import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



parser = argparse.ArgumentParser()
parser.add_argument('--metrics_folder', help='enter the test folder', required = True)
metrics_folder = parser.parse_args().metrics_folder
try:
    os.makedirs(metrics_folder)
except:
    pass

# Set style (same as your metrics plotting)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Read loss files
train_data = np.loadtxt('history/train_logs.txt')
val_data = np.loadtxt('history/val_logs.txt')

# Handle 1D vs 2D arrays
def get_x_y(data):
    if data.ndim == 1:
        return range(len(data)), data  # Use indices as x-axis
    else:
        return data[:, 0], data[:, 1]  # Use first column as x-axis

# Plot the data
train_x, train_y = get_x_y(train_data)
val_x, val_y = get_x_y(val_data)

plt.plot(train_x, train_y, marker='o', label='Train', linewidth=2)
plt.plot(val_x, val_y, marker='o', label='Validation', linewidth=2)

plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Loss', fontsize=18)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(metrics_folder + '/loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()


# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 4))

# File names
files = {
    'Train': ['history/acc_train.txt', 'history/rec_train.txt', 'history/prec_train.txt'],
    'Val': ['history/acc_val.txt', 'history/rec_val.txt', 'history/prec_val.txt']
}

metrics = ['Accuracy', 'Recall', 'Precision']

# Plot each metric
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    
    for split in ['Train', 'Val']:
        data = np.loadtxt(files[split][i])
        epochs = data[:, 0]
        scores = data[:, 1]
        plt.plot(epochs, scores, marker='o', label=split, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.title(f'{metric}', fontsize = 16)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(metrics_folder + '/acc_prec_rec.png', dpi=300, bbox_inches='tight')
plt.show()


def parse_roc_file(filename):
    """Parse ROC info file and return epoch data"""
    epochs_data = {}
    
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    # Split by lines and reconstruct complete entries
    lines = content.split('\n')
    current_entry = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        current_entry += " " + line if current_entry else line
        
        # Check if we have a complete entry (ends with ']')
        if current_entry.count('[') == current_entry.count(']') and current_entry.count('[') >= 2:
            try:
                # Find first space after epoch number
                first_space = current_entry.find(' ')
                epoch = int(current_entry[:first_space])
                
                # Find the two lists
                rest = current_entry[first_space:].strip()
                
                # Find where first list ends and second begins
                bracket_count = 0
                first_list_end = -1
                for i, char in enumerate(rest):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            first_list_end = i + 1
                            break
                
                if first_list_end > 0:
                    y_actual_str = rest[:first_list_end].strip()
                    scores_str = rest[first_list_end:].strip()
                    
                    # Clean and parse the lists
                    y_actual = parse_list_string(y_actual_str)
                    scores = parse_list_string(scores_str)
                    
                    # Convert to numpy arrays
                    y_actual = np.array(y_actual)
                    scores = np.array(scores)
                    
                    # Store or append to existing epoch data
                    if epoch in epochs_data:
                        epochs_data[epoch] = (
                            np.concatenate([epochs_data[epoch][0], y_actual]),
                            np.concatenate([epochs_data[epoch][1], scores])
                        )
                    else:
                        epochs_data[epoch] = (y_actual, scores)
                
                current_entry = ""
                
            except Exception as e:
                print(f"Error parsing entry: {current_entry[:100]}...")
                print(f"Error: {e}")
                current_entry = ""
    
    return epochs_data

def parse_list_string(list_str):
    """Parse list string, handling tensor notation and other formats"""
    # Remove tensor notation if present
    list_str = list_str.replace('tensor(', '').replace(')', '')
    
    # Try ast.literal_eval first
    try:
        return ast.literal_eval(list_str)
    except:
        # If that fails, try manual parsing
        # Remove brackets and split by comma
        list_str = list_str.strip('[]')
        items = [item.strip() for item in list_str.split(',')]
        
        # Try to convert to numbers
        result = []
        for item in items:
            try:
                # Try float first, then int
                if '.' in item:
                    result.append(float(item))
                else:
                    result.append(int(item))
            except:
                # Skip items that can't be converted
                continue
        
        return result

def plot_roc_from_file(filename, title_suffix=""):
    """Plot ROC curves from parsed file data"""
    epochs_data = parse_roc_file(filename)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs_data)))
    
    for i, (epoch, (y_actual, scores)) in enumerate(epochs_data.items()):
        try:
            # Ensure same length
            min_len = min(len(y_actual), len(scores))
            y_actual = y_actual[:min_len]
            scores = scores[:min_len]
            
            # print(f"Epoch {epoch}: y_actual shape={y_actual.shape}, scores shape={scores.shape}")
            
            # For MNIST multiclass ROC
            if scores.ndim == 1:
                # Single confidence scores - use binary classification for most frequent class
                unique_classes, counts = np.unique(y_actual, return_counts=True)
                most_frequent_class = unique_classes[np.argmax(counts)]
                y_binary = (y_actual == most_frequent_class).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, scores)
            else:
                # Multi-class probabilities - use micro-average
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_actual, classes=range(10))
                # Ensure scores match number of classes
                if scores.shape[1] != 10:
                    print(f"Warning: Expected 10 classes, got {scores.shape[1]}")
                    continue
                fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
            
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                     label=f'Epoch {epoch} (AUC = {roc_auc:.3f})')
        
        except Exception as e:
            print(f"Error processing epoch {epoch}: {e}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'ROC Curves{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(metrics_folder + '/' + f'roc_curves{title_suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


# Plot train and validation ROC curves
plot_roc_from_file('history/roc_inf_train.txt', '_train')
plot_roc_from_file('history/roc_inf_val.txt', '_val')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Load data from text file
y_true = []
y_scores = []
with open('history/roc_inf_test.txt', 'r') as f:
    content = f.read()
    
lines = content.strip().split('\n')
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line and line[0].isdigit():
        label = int(line.split()[0])
        bracket_start = line.find('[')
        if bracket_start != -1:
            scores_str = line[bracket_start+1:]
            while ']' not in scores_str and i+1 < len(lines):
                i += 1
                scores_str += ' ' + lines[i].strip()
            scores_str = scores_str.replace(']', '')
            scores = [float(x) for x in scores_str.split()]
            y_true.append(label)
            y_scores.append(scores)
    i += 1

y_true = np.array(y_true)
y_scores = np.array(y_scores)
y_true_bin = label_binarize(y_true, classes=range(10))

# Enhanced plotting
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

fpr, tpr, roc_auc = {}, {}, {}
auc_scores = []

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    auc_scores.append(roc_auc[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=line_styles[i], 
             linewidth=2.5, label=f'Class {i} (AUC = {roc_auc[i]:.3f})')

macro_auc = np.mean(auc_scores)
micro_auc = roc_auc_score(y_true_bin, y_scores, average='micro')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=22)
plt.ylabel('True Positive Rate', fontsize=22)
plt.title(f'ROC Curves - Macro AUC: {macro_auc:.3f}, Micro AUC: {micro_auc:.3f}', fontsize=22)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(metrics_folder + '/multiclass_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()