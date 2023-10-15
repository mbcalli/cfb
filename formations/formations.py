# Import data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from enum import Enum

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

class Side(Enum):
    OFFENSE = 1
    DEFENSE = 2

def get_relative_tracking(offense_df, defense_df, side: Side):
    # Create dataframe for each play
    game_and_play_id_df = offense_df[['playId', 'gameId']].drop_duplicates()

    rel_tracking = pd.DataFrame()

    for idx, row in game_and_play_id_df.iterrows():
        
        play_id = row['playId']
        game_id = row['gameId']
        
        if side == Side.OFFENSE:
            play_df = offense_df.query('playId == @play_id and gameId == @game_id')[['playId', 'playDirection', 'officialPosition', 'x', 'y']]
        else:
            play_df = defense_df.query('playId == @play_id and gameId == @game_id')[['playId', 'playDirection', 'officialPosition', 'x', 'y']]
        
        # Adjust postion relative to center
        try:
            center_record = offense_df.query('playId == @play_id and gameId == @game_id and officialPosition == "C"')[['playId', 'playDirection', 'officialPosition', 'x', 'y']].iloc[0]
        except IndexError:
            continue 
        
        C_x, C_y = center_record['x'], center_record['y']

        if play_df['playDirection'].iloc[0] == 'left':
            play_df['rel_x'] = C_x -play_df['x']
            play_df['rel_y'] = play_df['y'] - C_y
        else:
            play_df['rel_x'] = play_df['x'] - C_x
            play_df['rel_y'] = C_y - play_df['y']
            
        play_df = play_df[['playId', 'officialPosition', 'rel_x', 'rel_y']]
        
        # Sort by position and relative y to ensure consistent order
        play_df = play_df.sort_values(by=['officialPosition', 'rel_y'])[['officialPosition', 'rel_x', 'rel_y']]
        
        positions = play_df['officialPosition'].values
        
        # switch x and y
        xs = play_df['rel_y'].values
        ys = play_df['rel_x'].values
        personnel = ' '.join(positions)

        transformed_data = {
            'location': np.concatenate((xs, ys))
        }

        transformed_df = pd.DataFrame(transformed_data).T
        transformed_df['personnel'] = [personnel]
        transformed_df.columns = [f'x_{i}' for i in range(1, 12)] + [f'y_{i}' for i in range(1, 12)] + ['personnel']
        transformed_df = transformed_df.reset_index(drop=True)
        
        rel_tracking = pd.concat((rel_tracking, transformed_df))
        
    return rel_tracking

def plot_all_formations(rel_tracking, side: Side):
    xs = rel_tracking[[col for col in rel_tracking.columns if col.startswith('x')]].values
    ys = rel_tracking[[col for col in rel_tracking.columns if col.startswith('y')]].values

    plt.plot(xs, ys, 'ok', alpha=0.005)
    plt.xlabel('x')
    plt.ylabel('y')
    if side == Side.OFFENSE:
        plt.title('All offensive formations')
    elif side == Side.DEFENSE:
        plt.title('All defensive formations')
        
    plt.show()
        
def squish(x):
    if x > 0:
        return x
    return max(x**7, x)

def plot_cluster(df, model, label, ax, side: Side):
    
    # Get resulting df
    result = df.iloc[np.where(model.labels_ == label)]

    xs = result[[col for col in result.columns if col.startswith('x')]].values
    ys = result[[col for col in result.columns if col.startswith('y')]].values
    positions = result['personnel'].values[0].split(' ')

    for i in range(0, len(xs[0])):
        px = xs[:, i]
        
        # sqush ys closer to 0
        py = [squish(y) for y in ys[:, i]]
        ax.plot(px, py, 'o', alpha=0.1, color=(83/255, 173/255, 230/255))
        
        mean_x = np.mean(px)
        if side == Side.OFFENSE:
            mean_y = min(-0.3, np.mean(py))
        elif side == Side.DEFENSE:
            mean_y = np.mean(py)
        
        ax.text(mean_x, mean_y, positions[i], color='w', fontsize='medium')
        
    ax.set_facecolor((27/255, 107/255, 19/255))
    ax.axhline(0, color='w', alpha=0.2)
    
    # Plot hask marks
    if side == Side.OFFENSE:
        
        ax.set_xlim(-26.75, 26.75)
        ax.set_ylim(-8, 1)
        
        ax.axhline(-5, color='w', alpha=0.2)
        
        for i in range(1, 8):
            ax.plot([-6.1-0.5, -6.1+0.5], [-i, -i], '-', color='w', alpha=0.3)
            ax.plot([6.1-0.5, 6.1+0.5], [-i, -i], '-', color='w', alpha=0.3)
    elif side == Side.DEFENSE:
        
        ax.set_xlim(-26.75, 26.75)
        ax.set_ylim(-1, 20)
        
        ax.axhline(5, color='w', alpha=0.2)
        ax.axhline(10, color='w', alpha=0.2)
        ax.axhline(15, color='w', alpha=0.2)
        ax.axhline(20, color='w', alpha=0.2)
        for i in range(25):
            ax.plot([-6.1-0.5, -6.1+0.5], [i, i], '-', color='w', alpha=0.3)
            ax.plot([6.1-0.5, 6.1+0.5], [i, i], '-', color='w', alpha=0.3)
                    
def plot_model(specific_personnel_tracking, model, side: Side):
    _, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 7))

    for i, label in enumerate([x for x in np.unique(model.labels_) if x >= 0][:9]):
        
        row = i // 3
        col = i % 3
        
        plot_cluster(specific_personnel_tracking, model, label, axs[row, col], side)
        
    if isinstance(model, KMeans):
        model_type = 'K-Means'
    elif isinstance(model, DBSCAN):
        model_type = 'DBSCAN'
    else:
        model_type = 'Unknown'
        
    plt.suptitle(f'{model_type} ({side.name.lower()})')    
    plt.tight_layout()
    plt.show()
    
def train_k_means(specific_personnel_tracking, side: Side): 
    wcss = []

    for n in range(1, 20):
        km = KMeans(n_clusters=n, n_init=10, max_iter=300)
        km.fit(specific_personnel_tracking.drop('personnel', axis=1))
        wcss.append(km.inertia_)
        
    plt.plot(wcss)
    plt.axvline(6, color='r',linestyle='--', label='elbow (n=6)')
    plt.xlabel('n')
    plt.ylabel('WCSS')
    
    side_label = 'offense' if side == Side.OFFENSE else 'defense'
    
    plt.title(f'Elbow Method ({side_label})')
    plt.legend()
    plt.show()

    km = KMeans(n_clusters=6, n_init=3, max_iter=300)
    km.fit(specific_personnel_tracking.drop('personnel', axis=1))

    predictions = km.labels_
    
    return km, predictions

def train_dbscan(specific_personnel_tracking, side: Side):
    epses = np.linspace(0.5, 20, 39)
    entropies = []

    for eps in epses:
        db = DBSCAN(eps=eps)
        db.fit(specific_personnel_tracking.drop('personnel', axis=1))
        
        ps = {}
        for label in np.unique(db.labels_):
            ps[label] = np.sum(db.labels_ == label)

        entropy = 0
        for key, value in ps.items():
            p = value / len(db.labels_)
            entropy += p * np.log2(p)
        
        entropies.append(-entropy)

    plt.plot(epses, entropies)
    plt.xlabel('eps')
    plt.ylabel('Entropy')
    side_label = 'offense' if side == Side.OFFENSE else 'defense'
    plt.title(f'Entropy Plot ({side_label})')
    plt.grid()
    plt.show()
    
    db = DBSCAN(eps=epses[np.argmax(entropies)])
    db.fit(specific_personnel_tracking.drop('personnel', axis=1))
    
    return db, db.labels_