# this is wrong I think. logits over tokens are not computed in RM
# class RewardModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.gpt = GPT(config)  # Pretrained GPT model
#         self.classifier = nn.Linear(config.n_embd, 1)  # Output scalar reward

#     def forward(self, prompt_ids, response_ids):
#         # Concatenate prompt and response
#         input_ids = torch.cat([prompt_ids, response_ids], dim=1)
#         logits, _ = self.gpt(input_ids)
        
#         # Take the last hidden state for reward calculation
#         last_hidden = logits[:, -1, :]  # (B, n_embd)
#         reward = self.classifier(last_hidden)  # (B, 1)
#         return reward  # Return scalar reward


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT(config)
        self.reward_head = nn.Linear(config.n_embd, 1)  # Scalar reward output

    def forward(self, prompt_ids, response_ids):
        # Concatenate prompt and response to generate reward for the whole interaction
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        embeddings, _ = self.transformer(input_ids)
        rewards = self.reward_head(embeddings[:, -1, :])  # Use the last token embedding for reward
        return rewards


class KAnswerDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = row['prompt']
        answers = [row[f'answer_{i}'] for i in range(3)]  # 3 completions
        rankings = row['ranking']  # Ranking for answers: [2, 0, 1] means answer_2 > answer_0 > answer_1
        
        prompt_ids = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length, return_tensors='pt').squeeze()
        answer_ids = [self.tokenizer.encode(answer, truncation=True, max_length=self.max_length, return_tensors='pt').squeeze() for answer in answers]
        
        return {
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,  # list of 3 answer IDs
            "rankings": rankings  # list of rankings (e.g., [2, 0, 1])
        }

def reward_model_loss(reward_model, batch, device):
    prompt_ids = batch["prompt_ids"].to(device)
    answer_ids = batch["answer_ids"]  # List of 3 sets of response IDs
    rankings = batch["rankings"]  # The rankings for those answers (e.g., [2, 0, 1])

    # Compute rewards for all 3 responses
    rewards = []
    for i in range(3):
        response_ids = answer_ids[i].to(device)
        reward = reward_model(prompt_ids, response_ids)
        rewards.append(reward)
    
    rewards = torch.cat(rewards, dim=0)  # Shape: (3, 1)
    
    # Create all pairwise comparisons from rankings
    losses = []
    for i in range(3):
        for j in range(i + 1, 3):
            # i is better than j if rankings[i] < rankings[j]
            if rankings[i] < rankings[j]:
                reward_winner, reward_loser = rewards[i], rewards[j]
            else:
                reward_winner, reward_loser = rewards[j], rewards[i]
            
            # Compute the pairwise loss using log sigmoid
            loss = -torch.log(torch.sigmoid(reward_winner - reward_loser))
            losses.append(loss)
    
    # Average over all pairwise comparisons
    total_loss = torch.stack(losses).mean()
    return total_loss

def train_reward_model_with_k_comparisons(model, tokenizer, train_df, epochs=3, batch_size=2, lr=1e-5, device="cpu"):
    dataset = KAnswerDataset(train_df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = reward_model_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model


import pandas as pd

# Create toy dataset with a prompt and 3 answers, with ranking
toy_data = {
    "prompt": [
        "What is the capital of France?",
        "Tell me a joke.",
        "Explain the theory of relativity.",
    ],
    "answer_0": [
        "The capital of France is Paris.",
        "Why did the chicken cross the road? To get to the other side!",
        "The theory of relativity was developed by Einstein."
    ],
    "answer_1": [
        "France is a large country in Europe.",
        "I don't know any jokes.",
        "Relativity is a theory developed by Einstein."
    ],
    "answer_2": [
        "Paris is the capital of France.",
        "Here is a joke: Why did the chicken cross the road?",
        "Relativity explains how time and space are interconnected."
    ],
    "ranking": [
        [2, 0, 1],  # Answer 2 > Answer 0 > Answer 1
        [0, 2, 1],  # Answer 0 > Answer 2 > Answer 1
        [2, 0, 1]   # Answer 2 > Answer 0 > Answer 1
    ]
}

df_toy = pd.DataFrame(toy_data)
df_toy.to_csv("toy_k_comparisons_data.csv", index=False)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize models and tokenizer
    gpt_config = GPTConfig()
    reward_model = RewardModel(gpt_config)
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Load toy reward data
    df_reward = pd.read_csv("toy_k_comparisons_data.csv")
    
    # Train the reward model with K=3 comparisons
    trained_reward_model = train_reward_model_with_k_comparisons(reward_model, tokenizer, df_reward, epochs=3, batch_size=2, device=device)

#-----------------------------------------------------------------------------------------------------------

# A Simple PPO Implementation (High-Level Pseudocode)
class PPOAgent:
    def __init__(self, policy_model, reward_model, kl_coeff=0.1, lr=1e-5):
        self.policy_model = policy_model  # π_RL model
        self.reward_model = reward_model  # Reward Model
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)
        self.kl_coeff = kl_coeff  # KL penalty coefficient

    def compute_reward(self, prompt, response):
        # Get reward from the reward model
        reward = self.reward_model(prompt, response)
        return reward

    def compute_kl_penalty(self, old_policy_logits, new_policy_logits):
        # Compute the KL divergence between SFT model and RL model
        kl_div = F.kl_div(
            F.log_softmax(new_policy_logits, dim=-1),
            F.softmax(old_policy_logits, dim=-1),
            reduction="batchmean"
        )
        return kl_div

    def update_policy(self, prompt, old_policy_logits, old_value_estimates, actions, rewards):
        # Perform a forward pass through the RL policy model
        new_policy_logits, new_value_estimates = self.policy_model(prompt)

        # Compute PPO loss: reward + KL penalty
        policy_loss = -rewards + self.kl_coeff * self.compute_kl_penalty(old_policy_logits, new_policy_logits)

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

# Train the PPO agent with prompts and rewards from the reward model
def train_ppo(ppo_agent, train_data, sft_model):
    for batch in train_data:
        prompt = batch['prompt']
        old_policy_logits = sft_model(prompt)  # Old SFT model logits
        
        # Generate actions/responses from the current RL policy
        actions = ppo_agent.policy_model(prompt)
        
        # Compute rewards using the reward model
        rewards = ppo_agent.compute_reward(prompt, actions)
        
        # Update policy using PPO
        ppo_agent.update_policy(prompt, old_policy_logits, actions, rewards)



#-----------------------------------------------------------------------------------------------------------
# complete PPO implementation
class PPOAgent:
    def __init__(self, policy_model, reward_model, tokenizer, clip_epsilon=0.2, lr=3e-5, device="cpu"):
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.clip_epsilon = clip_epsilon
        self.device = device
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.reward_model.to(device)
        self.policy.to(device)
        self.reward_model.eval()  # Reward model should not be updated during PPO
        self.policy.train()

    def generate_response(self, prompt, max_length=50):
        """Generate response from the policy model."""
        self.policy.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            logits, _ = self.policy(input_ids)
            probs = F.softmax(logits, dim=-1)
            response_ids = torch.multinomial(probs[:, -1, :], num_samples=1).to(self.device)
            response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards using the reward model."""
        self.reward_model.eval()
        with torch.no_grad():
            prompt_ids = [self.tokenizer.encode(p, truncation=True, max_length=512, return_tensors='pt').squeeze() for p in prompts]
            response_ids = [self.tokenizer.encode(r, truncation=True, max_length=512, return_tensors='pt').squeeze() for r in responses]
            
            # Pad sequences
            prompt_ids_padded = nn.utils.rnn.pad_sequence(prompt_ids, batch_first=True).to(self.device)
            response_ids_padded = nn.utils.rnn.pad_sequence(response_ids, batch_first=True).to(self.device)
            
            rewards = self.reward_model(prompt_ids_padded, response_ids_padded).squeeze().cpu().numpy()
        return rewards
    
    def compute_log_probs(self, prompts, responses):
        """Compute log probabilities of actions (responses) given prompts."""
        self.policy.eval()
        log_probs = []
        for prompt, response in zip(prompts, responses):
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            logits, _ = self.policy(input_ids)
            probs = F.softmax(logits, dim=-1)
            
            response_ids = self.tokenizer.encode(response, return_tensors='pt').to(self.device)
            log_prob = 0
            for t in range(len(response_ids[0])):
                log_prob += torch.log(probs[:, t, response_ids[0][t]]) # extracts the prob of the specific selected token at position t in the response.
            log_probs.append(log_prob.item())
        
        return torch.tensor(log_probs, dtype=torch.float32, device=self.device)

    def ppo_step(self, prompts, old_responses, old_log_probs, rewards, advantages, epochs=4, batch_size=2):
        """Perform a PPO update step based on rewards and advantages."""
        dataset = list(zip(prompts, old_responses, old_log_probs, rewards, advantages))
        for epoch in range(epochs):
            batch_start = 0
            while batch_start < len(prompts):
                batch = dataset[batch_start:batch_start+batch_size]
                batch_prompts, batch_responses, batch_old_log_probs, batch_rewards, batch_advantages = zip(*batch)
                
                # Compute new log_probs for the current policy
                new_log_probs = self.compute_log_probs(batch_prompts, batch_responses)

                # Policy ratio
                ratio = torch.exp(new_log_probs - torch.tensor(batch_old_log_probs, device=self.device))

                # Clipping for the PPO objective
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                # Surrogate loss
                surrogate_loss = -torch.min(ratio * torch.tensor(batch_advantages, device=self.device), clipped_ratio * torch.tensor(batch_advantages, device=self.device)).mean()

                # Policy gradient step
                self.optimizer.zero_grad()
                surrogate_loss.backward()
                self.optimizer.step()

                batch_start += batch_size

            print(f"Epoch {epoch+1}/{epochs}, Loss: {surrogate_loss.item():.4f}")


def ppo_train(agent, prompts, epochs=3, batch_size=2):
    for epoch in range(epochs):
        # Generate responses from the policy
        responses = [agent.generate_response(prompt) for prompt in prompts]
        
        # Compute rewards using the reward model
        rewards = agent.compute_rewards(prompts, responses)
        
        # Normalize rewards to compute advantages
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = rewards  # In this simple case, we use rewards as advantages
        
        # Compute old log probabilities for the old policy
        old_log_probs = agent.compute_log_probs(prompts, responses).detach().cpu().numpy()
        
        # Perform PPO step
        agent.ppo_step(prompts, responses, old_log_probs, rewards, advantages)
        
        print(f"Epoch {epoch+1}/{epochs}, Average Reward: {rewards.mean():.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize models and tokenizer
    gpt_config = GPTConfig()
    gpt_model = GPT(gpt_config)
    reward_model = RewardModel(gpt_config)
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Load toy reward data
    df_reward = pd.read_csv("toy_reward_data.csv")
    
    # Train the reward model
    trained_reward_model = train_reward_model(reward_model, tokenizer, df_reward, epochs=3, batch_size=2, device=device)
    
    # Initialize PPO agent
    ppo_agent = PPOAgent(gpt_model, trained_reward_model, tokenizer, device=device)
    
    # Toy prompts
    prompts = ["What is the capital of France?", "Tell me a joke.", "Explain quantum mechanics."]
    
    # Train the policy using PPO
    ppo_train(ppo_agent, prompts, epochs=3, batch_size=2)
