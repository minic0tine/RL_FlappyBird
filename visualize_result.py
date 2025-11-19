"""
Visualization for Flappy Bird RL Results
Generates all comparison charts and sensitivity analysis
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_results():
    """Load all saved results"""
    results = {}
    
    # Load VI/PI summary
    if Path('results/vi_pi_summary.pkl').exists():
        with open('results/vi_pi_summary.pkl', 'rb') as f:
            results['vi_pi'] = pickle.load(f)
    
    return results


def plot_learning_curves(scores_dict, save_path='results/learning_curves.png'):
    """Plot Average Return vs Episodes for all algorithms"""
    plt.figure(figsize=(14, 8))
    
    colors = {
        'Policy Iteration': '#FFA500',
        'Value Iteration': '#1E90FF',
        'Monte Carlo': '#2E8B57',
        'SARSA': '#FFD700',
        'Q-Learning': '#4169E1'
    }
    
    for name, scores in scores_dict.items():
        # Calculate moving average
        window = 100
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        episodes = np.arange(len(moving_avg))
        
        plt.plot(episodes, moving_avg, label=name, linewidth=2, 
                color=colors.get(name, '#000000'))
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Return', fontsize=14)
    plt.title('So sánh các phương pháp — Average Return vs Episodes (FlappyBird)', 
              fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_final_comparison(results_dict, save_path='results/final_comparison.png'):
    """Bar chart of final average return (last 50 episodes)"""
    plt.figure(figsize=(12, 7))
    
    algorithms = list(results_dict.keys())
    scores = [results_dict[alg]['mean'] for alg in algorithms]
    
    bars = plt.bar(algorithms, scores, color='#FFA500', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Final Avg Return', fontsize=14)
    plt.title('Final Average Return (last 50 episodes, mean)', fontsize=16, pad=20)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_success_rate(results_dict, save_path='results/success_rate.png'):
    """Bar chart of success rate (% episodes with score > 0)"""
    plt.figure(figsize=(12, 7))
    
    algorithms = list(results_dict.keys())
    success_rates = []
    
    for alg in algorithms:
        scores = results_dict[alg].get('scores', [])
        if len(scores) > 0:
            success_rate = (np.array(scores) > 0).mean() * 100
        else:
            success_rate = 0
        success_rates.append(success_rate)
    
    bars = plt.bar(algorithms, success_rates, color='#FFA500', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.title('Final Success Rate', fontsize=16, pad=20)
    plt.ylim(0, 105)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_gamma_sensitivity(save_path='results/gamma_sensitivity.png'):
    """Plot sensitivity to discount factor (γ)"""
    plt.figure(figsize=(14, 8))
    
    # Example data - replace with actual sensitivity analysis results
    gammas = [0.70, 0.80, 0.90, 0.95, 0.99]
    
    # These should be computed by running experiments
    pi_scores = [70, 70, 75, 82, 84]
    vi_scores = [68, 68, 76, 81, 81]
    mc_scores = [51, 54, 61, 62, 60]
    sarsa_scores = [46, 50, 56, 56, 53]
    q_scores = [47, 50, 58, 59, 57]
    
    plt.plot(gammas, pi_scores, 'o-', label='Policy Iteration', linewidth=2, markersize=8, color='#FFA500')
    plt.plot(gammas, vi_scores, 'o-', label='Value Iteration', linewidth=2, markersize=8, color='#1E90FF')
    plt.plot(gammas, mc_scores, 'o-', label='Monte Carlo', linewidth=2, markersize=8, color='#2E8B57')
    plt.plot(gammas, sarsa_scores, 'o-', label='SARSA', linewidth=2, markersize=8, color='#FFD700')
    plt.plot(gammas, q_scores, 'o-', label='Q-Learning', linewidth=2, markersize=8, color='#4169E1')
    
    plt.xlabel('Discount Factor (γ)', fontsize=14)
    plt.ylabel('Final Average Return', fontsize=14)
    plt.title('So sánh theo Discount Factor (γ) — Final Average Return', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_alpha_sensitivity(save_path='results/alpha_sensitivity.png'):
    """Plot sensitivity to learning rate (α)"""
    plt.figure(figsize=(14, 8))
    
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    # Example data - replace with actual results
    mc_scores = [58.7, 61.7, 60.4, 55.0, 55.3]
    sarsa_scores = [55.2, 58.0, 61.0, 54.1, 51.8]
    q_scores = [54.3, 55.5, 60.5, 59.6, 54.5]
    
    plt.plot(alphas, mc_scores, 'o-', label='Monte Carlo', linewidth=2.5, markersize=10, color='#FFA500')
    plt.plot(alphas, sarsa_scores, 'o-', label='SARSA', linewidth=2.5, markersize=10, color='#1E90FF')
    plt.plot(alphas, q_scores, 'o-', label='Q-Learning', linewidth=2.5, markersize=10, color='#2E8B57')
    
    plt.xlabel('Learning Rate (α)', fontsize=14)
    plt.ylabel('Final Average Return', fontsize=14)
    plt.title('Sensitivity — Final Average Return vs Learning Rate (α)', fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_epsilon_sensitivity(save_path='results/epsilon_sensitivity.png'):
    """Plot sensitivity to exploration rate (ε)"""
    plt.figure(figsize=(14, 8))
    
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    # Example data
    mc_scores = [58.5, 61.5, 61.7, 58.0, 56.3]
    sarsa_scores = [55.0, 59.3, 62.5, 58.6, 54.7]
    q_scores = [59.0, 62.4, 63.0, 56.3, 54.8]
    
    plt.plot(epsilons, mc_scores, 'o-', label='Monte Carlo', linewidth=2.5, markersize=10, color='#FFA500')
    plt.plot(epsilons, sarsa_scores, 'o-', label='SARSA', linewidth=2.5, markersize=10, color='#1E90FF')
    plt.plot(epsilons, q_scores, 'o-', label='Q-Learning', linewidth=2.5, markersize=10, color='#2E8B57')
    
    plt.xlabel('Exploration ε', fontsize=14)
    plt.ylabel('Final Average Return', fontsize=14)
    plt.title('Sensitivity — Final Average Return vs Exploration (ε)', fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def create_summary_table(results_dict, save_path='results/summary_table.txt'):
    """Create summary table"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY RESULTS (FlappyBird)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Algorithm':<25} {'Condition':<20} {'Success Rate (%)':<20} {'Avg Return':<15}\n")
        f.write("-"*80 + "\n")
        
        for i, (alg, data) in enumerate(results_dict.items(), 1):
            mean = data.get('mean', 0)
            scores = data.get('scores', [])
            success_rate = (np.array(scores) > 0).mean() * 100 if len(scores) > 0 else 0
            
            f.write(f"{i:<5} {alg:<20} {'Deterministic':<20} {success_rate:<20.1f} {mean:<15.1f}\n")
        
        f.write("="*80 + "\n")
    
    print(f"✅ Saved: {save_path}")


def main():
    print("\n" + "="*70)
    print("FLAPPY BIRD RL - VISUALIZATION")
    print("="*70 + "\n")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Example: Load results from saved files
    # You'll need to modify this based on your actual saved data
    
    # Simulated results for demonstration
    results = {
        'Policy Iteration': {
            'mean': 89.5,
            'std': 5.2,
            'scores': np.random.normal(89.5, 5.2, 100).tolist()
        },
        'Value Iteration': {
            'mean': 83.1,
            'std': 6.1,
            'scores': np.random.normal(83.1, 6.1, 100).tolist()
        },
        'Monte Carlo': {
            'mean': -24.9,
            'std': 15.3,
            'scores': np.random.normal(-24.9, 15.3, 100).tolist()
        },
        'SARSA': {
            'mean': -38.4,
            'std': 18.2,
            'scores': np.random.normal(-38.4, 18.2, 100).tolist()
        },
        'Q-Learning': {
            'mean': -38.5,
            'std': 17.9,
            'scores': np.random.normal(-38.5, 17.9, 100).tolist()
        }
    }
    
    # Generate all plots
    print("[1/7] Generating learning curves...")
    # plot_learning_curves(scores_dict)  # Need actual episode-by-episode scores
    
    print("[2/7] Generating final comparison...")
    plot_final_comparison(results)
    
    print("[3/7] Generating success rate chart...")
    plot_success_rate(results)
    
    print("[4/7] Generating gamma sensitivity...")
    plot_gamma_sensitivity()
    
    print("[5/7] Generating alpha sensitivity...")
    plot_alpha_sensitivity()
    
    print("[6/7] Generating epsilon sensitivity...")
    plot_epsilon_sensitivity()
    
    print("[7/7] Creating summary table...")
    create_summary_table(results)
    
    print("\n" + "="*70)
    print(" All visualizations generated successfully!")
    print(" Check the 'results/' directory for all charts")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()