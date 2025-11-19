import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def create_gamma_table_dark():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.suptitle('Giả sử reward tương lai = 100', fontsize=14, fontweight='bold', y=0.95, color='white')

    data = [
        ['gamma', 'giá trị hội quy', 'ý nghĩa'],
        ['1.0', '100', 'coi tương lai quan trọng như hiện tại'],
        ['0.9', '90', 'tương lai quan trọng nhưng giảm nhẹ'],
        ['0.5', '50', 'tương lai chỉ bằng một nửa hiện tại'],
        ['0.0', '0', 'bỏ qua tương lai hoàn toàn']
    ]

    table = ax.table(cellText=data, cellLoc='left', loc='center', colWidths=[0.2, 0.25, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Header
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('white')
        cell.set_text_props(weight='bold', color='black')

    # Rows
    for i in range(1, 5):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor('dimgray')
            cell.set_text_props(color='white')
            if j == 0:
                cell.set_text_props(weight='bold', color='white')

    plt.savefig('results/table_gamma_dark.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_epsilon_table_dark():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.suptitle('Epsilon (ε) – Exploration', fontsize=14, fontweight='bold', y=0.95, color='white')

    data = [
        ['Giá trị ε', 'Ý nghĩa hành vi', 'Prob chọn random', 'Prob chọn action tốt nhất', 'Ghi chú'],
        ['1.0', 'Thăm dò hoàn toàn', '100%', '0%', 'Khám phá môi trường'],
        ['0.9', 'Thăm dò nhiều', '90%', '10%', 'Agent còn non'],
        ['0.7', 'Thăm dò mạnh', '70%', '30%', 'Bắt đầu học'],
        ['0.5', 'Cân bằng', '50%', '50%', 'Khám phá + khai thác'],
        ['0.3', 'Thăm dò ít', '30%', '70%', 'Agent ổn định'],
        ['0.1', 'Chủ yếu khai thác', '10%', '90%', 'Học tốt'],
        ['0.01', 'Gần như không thăm dò', '1%', '99%', 'Policy gần hội tụ'],
        ['0.0', 'Không thăm dò', '0%', '100%', 'Không nên dùng khi training']
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center', colWidths=[0.12,0.2,0.15,0.18,0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,2.2)

    # Header
    for i in range(5):
        cell = table[(0,i)]
        cell.set_facecolor('white')
        cell.set_text_props(weight='bold', color='black')

    # Rows
    for i in range(1, 9):
        for j in range(5):
            cell = table[(i,j)]
            cell.set_facecolor('dimgray')
            cell.set_text_props(color='white')
            if j == 0:
                cell.set_text_props(weight='bold', color='white')

    plt.savefig('results/table_epsilon_dark.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_alpha_table_dark():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.suptitle('Alpha (α) – Learning Rate', fontsize=14, fontweight='bold', y=0.95, color='white')

    data = [
        ['Alpha', 'Tốc độ học', 'Stability', 'Convergence', 'Khuyến nghị'],
        ['1.0', 'Cực nhanh', 'Rất kém', 'Không hội tụ', 'Không dùng'],
        ['0.5', 'Rất nhanh', 'Kém', 'Khó hội tụ', 'Chỉ dùng initial'],
        ['0.3', 'Nhanh', 'Trung bình', 'Chậm', 'OK cho quick test'],
        ['0.15', 'Vừa phải', 'Tốt', 'Vừa', 'Recommended'],
        ['0.1', 'Chậm', 'Tốt', 'Ổn định', 'Standard choice'],
        ['0.05', 'Rất chậm', 'Rất tốt', 'Chậm nhưng chắc', 'Conservative'],
        ['0.01', 'Cực chậm', 'Cực ổn', 'Rất chậm', 'Quá chậm']
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center', colWidths=[0.15,0.18,0.18,0.20,0.29])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,2.5)

    # Header
    for i in range(5):
        cell = table[(0,i)]
        cell.set_facecolor('white')
        cell.set_text_props(weight='bold', color='black')

    # Rows
    for i in range(1, 8):
        for j in range(5):
            cell = table[(i,j)]
            cell.set_facecolor('dimgray')
            cell.set_text_props(color='white')
            if j == 0:
                cell.set_text_props(weight='bold', color='white')

    plt.savefig('results/table_alpha_dark.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_recommendations_table_dark():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('tight')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.suptitle(' Recommended Hyperparameters cho Flappy Bird', fontsize=14, fontweight='bold', y=0.95, color='white')

    data = [
        ['Algorithm', 'Best γ', 'Best α', 'Best ε_min', 'Final Score'],
        ['Policy Iteration', '0.99', '-', '-', '89.5 '],
        ['Value Iteration', '0.99', '-', '-', '83.1 '],
        ['Monte Carlo', '0.95', '-', '0.1', '61.7'],
        ['SARSA', '0.95', '0.1', '0.1', '61.0'],
        ['Q-Learning', '0.95', '0.1', '0.1', '63.0']
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center', colWidths=[0.25,0.15,0.15,0.20,0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1,2.8)

    # Header
    for i in range(5):
        cell = table[(0,i)]
        cell.set_facecolor('white')
        cell.set_text_props(weight='bold', color='black')

    # Rows
    for i in range(1, 6):
        for j in range(5):
            cell = table[(i,j)]
            cell.set_facecolor('dimgray')
            cell.set_text_props(color='white')
            if j == 0 or j == 4:
                cell.set_text_props(weight='bold', color='white')

    plt.savefig('results/table_recommendations_dark.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def main():
    os.makedirs('results', exist_ok=True)
    print("Generating all dark tables...")
    create_gamma_table_dark()
    create_epsilon_table_dark()
    create_alpha_table_dark()
    create_recommendations_table_dark()
    print(" All dark tables saved in results/")

if __name__ == "__main__":
    main()
