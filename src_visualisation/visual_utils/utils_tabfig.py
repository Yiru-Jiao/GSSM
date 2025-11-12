'''
This file contains the functions to create the tables and figures
'''

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_safety_evaluation.validation_utils.utils_eval_metrics import *


cmap = mpl.colors.LinearSegmentedColormap.from_list('cet_cbl2', cc.CET_CBL2[::-1])
cmap_diverge = mpl.colors.LinearSegmentedColormap.from_list('cet_cbd1', cc.CET_CBD1)
cmap_cycle = mpl.colors.LinearSegmentedColormap.from_list('cet_cbc2', cc.CET_CBC2)
cmap_cw = mpl.colors.LinearSegmentedColormap.from_list('cet_cbtd1', cc.CET_CBTD1)

category = {'SafeBaseline': 'Safe interactions',
            'NearCrash': 'Near-crashes',
            'Crash': 'Crashes',
            'NearCrash-NearCrash': 'Near-crashes',
            'SecondaryNearCrash': 'Near-crashes',
            'NearCrash-CrashRelevant': 'Near-crashes',
            'NearCrash-Crash': 'Crashes',
            'Crash-Crash': 'Crashes',
            'Crash-NearCrash': 'Crashes',
            'CrashRelevant-NearCrash': 'Near-crashes',
            'SecondaryCrash': 'Crashes',
            'NearCrash-OtherConflict': 'Near-crashes',
            'Crash-OtherConflict': 'Crashes',
            'CrashRelevant-Crash': 'Crashes',
            'Crash-CrashRelevant': 'Crashes'}

conflict_type = {'leading': 'Leading',
                 'following': 'Following',
                 'adjacent_lane': 'Adjacent lane',
                 'merging': 'Merging',
                 'turning_into_opposite': 'Crossing/turning',
                 'turning_into_parallel': 'Crossing/turning',
                 'turning_across_opposite': 'Crossing/turning',
                 'turning_across_parallel': 'Crossing/turning',
                 'intersection_crossing': 'Crossing/turning',
                 'pedestrian': 'Pedestrian/cyclist',
                 'cyclist': 'Pedestrian/cyclist',
                 'animal': 'Animal',
                 'oncoming': 'Oncoming',
                 'obstacle': 'Shapeless obstacle',
                 'single': 'Single',
                 'parked': 'Parked',
                 'unknown': 'Unknown',
                 'none': 'None',
                 'nan': 'None'}


def remove_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)
    ax.set_frame_on(False)


def light_color(color, alpha=0.5):
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)
    return tuple([c + (1 - c) * (1 - alpha) for c in color])
    

def highlight_col(col, highlight_type='max', scientific_notation=False, involve_second=False):
    if np.mean(col)>0.01:
        col = np.round(col.values, 3)
    else:
        col = col.values
    if highlight_type == 'max':
        sorted_col = np.sort(col[~np.isnan(col)])[::-1]
    elif highlight_type == 'min':
        sorted_col = np.sort(col[~np.isnan(col)])
    else:
        raise ValueError("highlight_type must be either 'max' or 'min'.")
    
    if len(sorted_col) >= 2:
        is_extreme = col==sorted_col[0]
        is_second_extreme = col==sorted_col[1]
    elif len(sorted_col) == 1:
        is_extreme = col==sorted_col[0]
        is_second_extreme = np.zeros(len(col), dtype=bool)
    else:
        is_extreme = np.zeros(len(col), dtype=bool)
        is_second_extreme = np.zeros(len(col), dtype=bool)
    
    styled_col = []
    if scientific_notation and np.nanmean(col)<=0.01:
        for i, v in enumerate(col):
            if is_extreme[i]:
                styled_col.append(f'\\textbf{{\\underline{{{v:.{scientific_notation}E}}}}}'.replace('E-0','E-'))
            elif is_second_extreme[i] and involve_second:
                styled_col.append(f'\\textbf{{{v:.{scientific_notation}E}}}'.replace('E-0','E-'))
            else:
                styled_col.append(f'{v:.{scientific_notation}E}'.replace('E-0','E-'))
    else:
        for i, v in enumerate(col):
            if is_extreme[i]:
                styled_col.append(f'\\textbf{{\\underline{{{v:.3f}}}}}')
            elif is_second_extreme[i] and involve_second:
                styled_col.append(f'\\textbf{{{v:.3f}}}')
            else:
                styled_col.append(f'{v:.3f}')
    return styled_col


def highlight(df, max_cols=[], min_cols=[], scientific_notation=False, involve_second=False):
    for col in max_cols:
        df[col] = highlight_col(df[col], 'max', scientific_notation, involve_second)
    for col in min_cols:
        df[col] = highlight_col(df[col], 'min', scientific_notation, involve_second)
    return df


def to_grayscale(fig):
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    grayscale_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    fig_gray, ax_gray = plt.subplots(figsize=(fig.get_size_inches()), dpi=fig.dpi)
    ax_gray.imshow(grayscale_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    ax_gray.axis('off')  # Turn off the axis
    ax_gray.set_title('Grayscale plot')
    return fig_gray


def read_statistics(path_raw, database, by):
    if database == 'IHME':
        data = pd.read_csv(path_raw + f'StatisticsAccidents/{database}/{by}.csv', thousands=',')
    if database == 'BRON':
        data = pd.read_csv(path_raw + f'StatisticsAccidents/{database}/{by}.csv', thousands=',')
    elif database == 'UNECE':
        data = pd.read_csv(path_raw + f'StatisticsAccidents/{database}/{by}.csv')
    elif database == 'NHTSA':
        data = pd.read_csv(path_raw + f'StatisticsAccidents/{database}/{by}.csv', thousands=',')
        # data = data[data['Year']<2022]
        if by == 'weather':
            data['Number of accidents'] = data['Fatal crashes'] + data['Injury-only crashes']
            columns = ['Fatal crashes', 'Injury-only crashes', 'Number of accidents']
            data = data.groupby('Weather')[columns].sum()
            data.loc['Not adverse'] = data.loc['No Adverse Atmospheric Conditions/Clear/Cloudy'] + data.loc['Cloudy']
            data.loc['Unknown'] = data.loc['Other'] + data.loc['Unknown/Not Reported']
            data.loc['Rainy'] = data.loc['Rain (Mist)'] + data.loc['Sleet, Hail (Freezing Rain or Drizzle)'] + data.loc['Freezing Rain or Drizzle']
            data.loc['Foggy'] = data.loc['Fog, Smog, Smoke']
            data.loc['Snowy'] = data.loc['Snow'] + data.loc['Blowing Snow']
            data.loc['Windy'] = data.loc['Blowing Sand, Soil, Dirt'] + data.loc['Severe Crosswinds']
            data.loc['Adverse'] = data.loc['Rainy'] + data.loc['Foggy'] + data.loc['Snowy'] + data.loc['Windy']
            data.loc['All'] = data.loc['Not adverse'] + data.loc['Adverse']
            data = data.loc[['All', 'Not adverse', 'Adverse', 'Rainy', 'Foggy', 'Snowy', 'Windy', 'Unknown']].reset_index()
        elif by == 'light':
            data['Number of accidents'] = data['Fatal crashes'] + data['Injury-only crashes']
            columns = ['Fatal crashes', 'Injury-only crashes', 'Number of accidents']
            data = data.groupby('Light Condition')[columns].sum()
            data.loc['Dawn/dusk'] = data.loc['Dawn'] + data.loc['Dusk']
            data.loc['Unknown'] = data.loc['Other'] + data.loc['Not Reported']
            data.loc['Dark'] = data.loc['Dark - Not Lighted'] + data.loc['Dark - Lighted'] + data.loc['Dark - Unknown Lighting']
            data.loc['All'] = data.loc['Dark'] + data.loc['Daylight'] + data.loc['Dawn/dusk'] + data.loc['Unknown']
            data = data.loc[['All', 'Dark', 'Dark - Not Lighted', 'Dark - Lighted', 'Dark - Unknown Lighting', 'Daylight', 'Dawn/dusk', 'Unknown']].reset_index()
    return data


def location_type_nl(ax, path_data):
    ax_all = ax
    remove_box(ax_all)
    data_intersection = read_statistics(path_data, 'BRON','intersection_type')
    data_intersection = data_intersection[['Location','Type of crash']+np.arange(2021,2024).astype(str).tolist()]
    road_user_types = ['Pedestrian','Frontal','Lateral','Rear-end']
    data_intersection = data_intersection[data_intersection['Type of crash'].isin(road_user_types)]
    at_intersection = data_intersection[data_intersection['Location']=='Intersection'].set_index('Type of crash').drop(columns=['Location']).T
    at_intersection = at_intersection.T.sum(axis=1).loc[road_user_types]
    not_at_intersection = data_intersection[data_intersection['Location']=='Road section'].set_index('Type of crash').drop(columns=['Location']).T
    not_at_intersection = not_at_intersection.T.sum(axis=1).loc[road_user_types]
    data_intersection['Sum'] = data_intersection[np.arange(2021,2024).astype(str)].sum(axis=1)
    data_intersection = data_intersection.groupby('Location')['Sum'].sum()

    ax_at_int = ax.inset_axes([-0.1, -0.05, 0.6, 1], xlim=(0, 1), ylim=(-0.5, 0.5))
    ax_not_at = ax.inset_axes([0.45, -0.05, 0.6, 1], xlim=(0, 1), ylim=(-0.5, 0.5))

    colors = cmap(np.linspace(0.35, 0.75, len(at_intersection)))
    colors[0] = cmap(0.15)
    patches, _, autopct = ax_at_int.pie(at_intersection, autopct='%1.1f%%', normalize=True,
                                        pctdistance=0.65, textprops={'color': 'w'}, startangle=180,
                                        colors=colors)
    autopct[-1].set_y(autopct[-1].get_position()[1]+0.1)
    patches, _, autopct = ax_not_at.pie(not_at_intersection, autopct='%1.1f%%', normalize=True,
                                        pctdistance=0.65, textprops={'color': 'w'}, startangle=180,
                                        colors=colors)
    autopct[-1].set_y(autopct[-1].get_position()[1]+0.1)
    ax_not_at.legend(patches, road_user_types, loc='center left', bbox_to_anchor=(1., 0.5),
                     ncol=1, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)
    
    int_rate = data_intersection.loc['Intersection'].sum()/data_intersection.loc[['Intersection','Road section']].sum()
    xlim = ax_at_int.get_xlim()
    ylim = ax_at_int.get_ylim()
    ax_at_int.text(xlim[0]+0.5*(xlim[1]-xlim[0]), ylim[1]*1.05,
                   f'At intersection ({int_rate*100:.1f}%)', ha='center', va='top')
    xlim = ax_not_at.get_xlim()
    ylim = ax_not_at.get_ylim()
    ax_not_at.text(xlim[0]+0.5*(xlim[1]-xlim[0]), ylim[1]*1.05,
                   f'Not at intersection ({(1-int_rate)*100:.1f}%)', ha='center', va='top')


def location_type_us(ax, path_data):
    ax_all = ax
    remove_box(ax_all)

    at_intersection = read_statistics(path_data, 'NHTSA','at_intersection')
    at_intersection['Number of accidents'] = at_intersection['Fatal crashes'] + at_intersection['Injury-only crashes']
    at_intersection = at_intersection[['Year','Crash type','Number of accidents']]
    at_intersection = at_intersection[at_intersection['Year'].isin(np.arange(2021,2024))]
    road_user_types = ['Sideswipe', 'Head-On', 'Angle', 'Rear-End']
    at_intersection = at_intersection[at_intersection['Crash type'].isin(road_user_types)]
    at_intersection = at_intersection.groupby('Crash type')['Number of accidents'].sum().loc[road_user_types]
    not_at_intersection = read_statistics(path_data, 'NHTSA','not_at_intersection')
    not_at_intersection['Number of accidents'] = not_at_intersection['Fatal crashes'] + not_at_intersection['Injury-only crashes']
    not_at_intersection = not_at_intersection[['Year','Crash type','Number of accidents']]
    not_at_intersection = not_at_intersection[not_at_intersection['Year'].isin(np.arange(2021,2024))]
    not_at_intersection = not_at_intersection[not_at_intersection['Crash type'].isin(road_user_types)]
    not_at_intersection = not_at_intersection.groupby('Crash type')['Number of accidents'].sum().loc[road_user_types]

    colors = cmap(np.linspace(0.35, 0.75, len(at_intersection)))
    ax_at_int = ax.inset_axes([-0.1, -0.05, 0.6, 1], xlim=(0, 1), ylim=(-0.5, 0.5))
    ax_not_at = ax.inset_axes([0.45, -0.05, 0.6, 1], xlim=(0, 1), ylim=(-0.5, 0.5))
    patches, _, autopct = ax_at_int.pie(at_intersection, autopct='%1.1f%%', normalize=True,
                                        pctdistance=0.65, textprops={'color': 'w'}, startangle=180,
                                        colors=colors)
    patches, _, autopct = ax_not_at.pie(not_at_intersection, autopct='%1.1f%%', normalize=True,
                                        pctdistance=0.65, textprops={'color': 'w'}, startangle=180,
                                        colors=colors)
    ax_not_at.legend(patches, ['Sideswipe', 'Head-on', 'Angle', 'Rear-end'], loc='center left', bbox_to_anchor=(1., 0.5),
                     ncol=1, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)

    int_rate = at_intersection.sum()/(at_intersection.sum() + not_at_intersection.sum())
    xlim = ax_at_int.get_xlim()
    ylim = ax_at_int.get_ylim()
    ax_at_int.text(xlim[0]+0.5*(xlim[1]-xlim[0]), ylim[1]*1.05,
                   f'At intersection ({int_rate*100:.1f}%)', ha='center', va='top')
    xlim = ax_not_at.get_xlim()
    ylim = ax_not_at.get_ylim()
    ax_not_at.text(xlim[0]+0.5*(xlim[1]-xlim[0]), ylim[1]*1.05,
                   f'Not at intersection ({(1-int_rate)*100:.1f}%)', ha='center', va='top')
    

def reconstruction_error(data_ego, data_sur, event_type, figsize=(7.05,4.)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    for row, target, data in zip(range(2), ['Subject', 'Object'], [data_ego, data_sur]):
        if target=='Subject':
            var_list = {'Speed': ['m/s', 'v_ekf','speed_comp'], 'Acceleration': ['m/s$^2$', 'acc_ekf','acc_lon']}
        else:
            var_list = {'Displacement': ['m'], 'Speed': ['m/s', 'v_ekf','speed_comp']}

        for col, key, values in zip(range(2), var_list.keys(), var_list.values()):
            if key == 'Displacement':
                error = ((data['x_ekf']-data['x'])**2 + (data['y_ekf']-data['y'])**2)**0.5
            else:
                error = data[values[1]] - data[values[2]]
            
            mean, std = error.mean(), error.std()
            limits = [mean - 3*std, mean + 3*std] if key != 'Displacement' else [0, mean + 3*std]
            _ = axes[row, col].hist(error, alpha=0.75, bins=np.linspace(limits[0], limits[1], 30))
            axes[row, col].text(0.95, 0.95, f'mean={mean:.4f}\nstd={std:.4f}', ha='right', va='top', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'{target} {key} error ({values[0]})')
    _ = fig.suptitle(f'{event_type} event reconstruction error distribution')
    return fig


def stacked_bar(ax, values, labels, colors=None, direction='vertical', alpha=1, hatches=None):
    # Adjust values to make sure the sum is 100
    percentages = np.round(values/np.sum(values), 3)
    if percentages.sum() < 1:
        gap = 1 - percentages.sum()
        percentages[np.argmax(values)] += gap
        percentages = np.round(percentages, 3)
    if percentages.sum() > 1:
        gap = percentages.sum() - 1
        percentages[np.argmax(values)] -= gap
        percentages = np.round(percentages, 3)
    
    # Plot stacked bars
    base = 0
    txt = []
    if colors is None:
        colors = cmap(np.linspace(0.15,0.85,len(values)))
    if hatches is None:
        hatches = ['' for _ in range(len(values))]
    for percentage, label, color, hatch in zip(percentages, labels, colors, hatches):
        if direction=='vertical':
            ax.bar(0, percentage, bottom=base, color=color, label=label, width=1, alpha=alpha, hatch=hatch)
            txt.append(ax.text(0, base+percentage/2, f'{percentage*100:.1f}%', ha='center', va='center', color='w'))
        if direction=='horizontal':
            ax.barh(0, percentage, left=base, color=color, label=label, height=1, alpha=alpha, hatch=hatch)
            txt.append(ax.text(base+percentage/2, 0, f'{percentage*100:.1f}%', ha='center', va='center', color='w'))
        base += percentage
    if direction=='vertical':
        ax.set_ylim(0,1)
        ax.set_yticks([])
    if direction=='horizontal':
        ax.set_xlim(0,1)
        ax.set_xticks([])

    # Adjust text position to avoid overlap
    for i in range(1, len(txt)):
        if direction=='vertical':
            if txt[i].get_position()[1] < txt[i-1].get_position()[1]+0.05:
                txt[i].set_position((txt[i].get_position()[0], txt[i-1].get_position()[1]+0.05))
        if direction=='horizontal':
            if txt[i].get_position()[0] < txt[i-1].get_position()[0]+0.05:
                txt[i].set_position((txt[i-1].get_position()[0]+0.05, txt[i].get_position()[1]))


def global_deaths_over_years(ax, path_raw):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlim(2000,2022)
    ax.set_xticks([2001, 2006, 2011, 2016, 2021, 2022])
    ax.set_xticklabels(['2001', '2006', '2011', '2016', '2021', ''])

    data = read_statistics(path_raw, 'IHME', 'Deaths').sort_values('year')
    data = data[(data['measure']=='Deaths')&(data['year']>=2001)&(data['year']<2022)]

    lower_upper = data[['lower','upper']].copy()
    lower_upper['Lower'] = data['val'] - data['lower']
    lower_upper['Upper'] = data['upper'] - data['val']
    lower_upper = lower_upper[['Lower','Upper']].values.T
    ax.errorbar(data['year'], data['val'], yerr=lower_upper, linewidth=0.5, zorder=-10,
                color=light_color(cmap(0.85),0.85), capsize=2, capthick=0.5, elinewidth=0.5)
    ax.plot(data['year'], data['val'], 'o-', markersize=3, color=cmap(0.85), label='Global deaths',
            markeredgecolor=cmap(0.85), markerfacecolor=light_color(cmap(0.85),0.5), linewidth=0.5, markeredgewidth=0.5)
    ax.yaxis.set_major_formatter(lambda x, pos: '{:,.1f}'.format(x/1e6))
    ax.set_ylim(ax.get_ylim()[0]*0.8, ax.get_ylim()[1])
    ax.set_yticks([1e6, 1.2e6, ax.get_ylim()[1]])
    ax.set_yticklabels(['1.0', '1.2', ''])
    ax.text(2000, ax.get_ylim()[1]*1.005, 'in millions', ha='center', va='bottom')
    ax.legend(frameon=False, ncol=1, columnspacing=0.5, handletextpad=0.5)


def unece_accident_reduction(ax, path_raw):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlim(2000,2022)
    ax.set_xticks([2001, 2006, 2011, 2016, 2021, 2022])
    ax.set_xticklabels(['2001', '2006', '2011', '2016', '2021', ''])

    data_unece = read_statistics(path_raw, 'UNECE','location')
    data_unece = data_unece[(data_unece['Year']>=2001)&(data_unece['Year']<2022)]

    data_unece = data_unece[~data_unece['Number of accidents'].isna()]
    complete = data_unece.groupby(['Country','Location'])['Year'].count()
    complete = complete[complete>=complete.max()].index.get_level_values(0).unique()
    print(f'{len(complete)} countries have complete data and are used to make plot.')
    data_unece = data_unece[data_unece['Country'].isin(complete)]

    data_unece = data_unece.groupby(['Location','Year'])['Number of accidents'].sum()
    data_unece = data_unece.reset_index().sort_values('Year').set_index('Year')
    motorways = data_unece[data_unece['Location']=='Motorways']['Number of accidents']
    inside_builtup = data_unece[data_unece['Location']=='Inside built-up areas']['Number of accidents']
    outside_builtup = data_unece[data_unece['Location']=='Outside built-up areas']['Number of accidents']
    ax.bar(outside_builtup.index, outside_builtup, label='Outside built-up areas', color=cmap(0.65), width=0.8)
    ax.bar(inside_builtup.index, inside_builtup, bottom=outside_builtup, label='Inside built-up areas',
           color=cmap(0.4), width=0.8)
    ax.bar(motorways.index, motorways, bottom=inside_builtup+outside_builtup, label='Motorways',
           color=cmap(0.15), width=0.8)

    ax.set_ylim(0, 2e6)
    ax.set_yticks([0, 0.5e6, 1e6, 1.5e6, 2e6])
    ax.set_yticklabels(['0', '0.5', '1.0', '1.5', ''])
    ax.text(2000, 2e6*1.005, 'in millions', ha='center', va='bottom')
    ax.set_ylabel('Number of accidents', labelpad=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1.14), ncol=1, 
              frameon=False, columnspacing=0.5, handletextpad=0.5, handlelength=0.75)
    
    data_unece = pd.pivot_table(data_unece, values='Number of accidents', index='Year', columns='Location')
    data_unece['Ratio_Motorways'] = data_unece['Motorways']/data_unece.sum(axis=1)
    ax = ax.inset_axes([0., 0., 1, 0.58], xlim=(2000,2022), ylim=(0.04,0.06))
    ax.patch.set_alpha(0.0)
    ax.set_xticks([])
    for spine in ['top', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.yaxis.tick_right()
    ax.set_yticks([0.04, 0.05, 0.06])
    ax.set_yticklabels(['4%', '5%', '6%'])
    ax.yaxis.set_label_coords(1.08, 0.5)
    ax.set_ylabel('Motorway percentage', rotation=270, va='bottom')
    ax.plot(data_unece.index, data_unece['Ratio_Motorways'], 'o-', 
            markersize=2, color=cmap(0.15), linewidth=0.75)
    

def read_meta(path_processed, path_result):
    meta_all = pd.read_csv(path_processed + 'SHRP2/metadata_birdseye.csv').set_index('event_id')
    meta_all.loc[meta_all['event_category']=='Crash', 'severity_first'] = 3
    meta_all.loc[meta_all['event_category']=='Crash', 'severity_second'] = 0
    meta_all.loc[meta_all['event_category']=='NearCrash', 'severity_first'] = 2
    meta_all.loc[meta_all['event_category']=='NearCrash', 'severity_second'] = 0
    secondary_events = ['SecondaryCrash', 'SecondaryNearCrash']
    meta_all.loc[meta_all['event_category'].isin(secondary_events), 'severity_first'] = 0
    meta_all.loc[meta_all['event_category']=='SecondaryCrash', 'severity_second'] = 3
    meta_all.loc[meta_all['event_category']=='SecondaryNearCrash', 'severity_second'] = 2
    connected_events = ['Crash-Crash', 'Crash-NearCrash', 'NearCrash-Crash', 'NearCrash-NearCrash',
                        'NearCrash-CrashRelevant', 'CrashRelevant-NearCrash', 
                        'NearCrash-OtherConflict', 'Crash-OtherConflict',
                        'CrashRelevant-Crash', 'Crash-CrashRelevant']
    condition = meta_all['event_category'].isin(connected_events)
    first = meta_all['event_category'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    second = meta_all['event_category'].apply(lambda x: x.split('-')[1] if '-' in x else x)
    meta_all.loc[condition&(first=='Crash'), 'severity_first'] = 3
    meta_all.loc[condition&(first=='NearCrash'), 'severity_first'] = 2
    meta_all.loc[condition&(meta_all['severity_first'].isna()), 'severity_first'] = 1
    meta_all.loc[condition&(second=='Crash'), 'severity_second'] = 3
    meta_all.loc[condition&(second=='NearCrash'), 'severity_second'] = 2
    meta_all.loc[condition&(meta_all['severity_second'].isna()), 'severity_second'] = 1

    meta_all.loc[meta_all['event_category']=='SafeBaseline', 'conflict'] = 'none'
    meta_all.loc[(meta_all['severity_first']<0.5)&(meta_all['severity_second']<0.5), 'conflict'] = 'none'
    condition = meta_all['severity_first']>=meta_all['severity_second']
    meta_all.loc[condition, 'conflict'] = meta_all.loc[condition, 'first']
    condition = meta_all['severity_second']>meta_all['severity_first']
    meta_all.loc[condition, 'conflict'] = meta_all.loc[condition, 'second']
    meta_all['conflict'] = [conflict_type[c] for c in meta_all['conflict'].values]

    meta_reconstructed = meta_all[(meta_all['ego_reconstructed'].astype(bool))&
                                  (meta_all['surrounding_reconstructed'].astype(bool))]

    event_categories = sorted(os.listdir(path_result + 'EventData/'))
    meta_events = pd.concat([pd.read_csv(path_result + 'EventData/' + f'{event_cat}/event_meta.csv') for event_cat in event_categories])
    meta_events = meta_events[meta_events['duration_enough']&(meta_events['conflict']!='none')].set_index('event_id')
    meta_events[['severity_first','severity_second','conflict']] = meta_reconstructed.loc[meta_events.index, ['severity_first','severity_second','conflict']].values

    environment = pd.concat([pd.read_csv(path_result + 'EventData/' + f'{event_cat}/environment.csv') for event_cat in event_categories])
    environment = environment.set_index('event_id').loc[meta_events.index]

    return meta_all, meta_reconstructed, meta_events, environment


def get_data(meta_all, meta_reconstructed, meta_events):
    counts = []
    all_events = meta_all.groupby('event_category').size().rename('Recorded')
    counts.append(all_events)

    reconstructed_events = meta_reconstructed.groupby('event_category').size().rename('Reconstructed')
    counts.append(reconstructed_events)

    assessible_events = meta_events.groupby('event_category').size().rename('In test set')
    counts.append(assessible_events)

    counts = pd.concat(counts, axis=1).fillna(0)
    counts['category'] = [category[c] for c in counts.index]
    counts = counts.groupby('category')[['Recorded', 'Reconstructed', 'In test set']].sum()
    counts = counts.astype(int)

    all_conflicts = meta_all[meta_all['event_category']!='SafeBaseline'].groupby('conflict').size().rename('Count')
    reconstructed_conflicts = meta_reconstructed[meta_reconstructed['event_category']!='SafeBaseline'].groupby('conflict').size().rename('Count')
    test_conflicts = meta_events[meta_events['event_category']!='SafeBaseline'].groupby('conflict').size().rename('Count')

    return counts, all_conflicts, reconstructed_conflicts, test_conflicts


def draw_SHRP2(path_processed, path_result, figsize=(7.05,3.8)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios':[1.3, 1]}, constrained_layout=True)
    remove_box(axes[0])
    remove_box(axes[1])

    meta_all, meta_reconstructed, meta_events, environment = read_meta(path_processed, path_result)
    counts, all_conflicts, reconstructed_conflicts, test_conflicts = get_data(meta_all, meta_reconstructed, meta_events)
    all_conflicts = all_conflicts.sort_values(ascending=False)
    reconstructed_conflicts_order = []
    test_conflicts_order = []
    for ctype in all_conflicts.index:
        if ctype in reconstructed_conflicts.index:
            reconstructed_conflicts_order.append(ctype)
        if ctype in test_conflicts.index:
            test_conflicts_order.append(ctype)
    reconstructed_conflicts = reconstructed_conflicts.loc[reconstructed_conflicts_order]
    test_conflicts = test_conflicts.loc[test_conflicts_order]

    ax_data = axes[0].inset_axes([0, 0, 0.35, 1])
    remove_box(ax_data)
    ax_beyond = ax_data.inset_axes([0, 0.9, 1, 0.1])
    ax_within = ax_data.inset_axes([0, 0, 1, 0.865])
    ax_within.spines['right'].set_visible(False)
    ax_beyond.spines['top'].set_visible(False)
    ax_beyond.spines['right'].set_visible(False)
    threshold = 13000
    beyond_bottom = counts['Recorded'].sum() - threshold/0.865*0.1
    ax_within.set_ylim(0, threshold)
    ax_within.set_ylabel('Number of events', labelpad=1)
    ax_beyond.set_ylim(beyond_bottom, counts['Recorded'].sum())
    # Draw the diagonal lines to indicate broken y axis
    ax_within.spines['top'].set_visible(False)
    ax_beyond.spines['bottom'].set_visible(False)
    ax_beyond.set_xticks([])
    kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=5,
                linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)
    ax_within.plot([0], [1], transform=ax_within.transAxes, **kwargs)
    ax_beyond.plot([0], [0], transform=ax_beyond.transAxes, **kwargs)

    ax_conflicts = axes[0].inset_axes([0.45, 0, 0.55, 1])
    ax_conflicts.tick_params(axis='both', which='both', direction='in')
    ax_conflicts.spines['top'].set_visible(False)
    ax_conflicts.spines['right'].set_visible(False)
    colorlist = np.concatenate([np.linspace(0.15,0.6,5),
                                np.linspace(0.65,0.85,len(all_conflicts)-5)])
    colors = {key: cmap(colorlist[i]) for i, key in enumerate(all_conflicts.index)}

    # Recorded
    base = 0
    ax_within.text(0, -200, 'Recorded', ha='center', va='top')
    for cat, color in zip(['Crashes','Near-crashes','Safe interactions'], cmap([0.65, 0.4, 0.15])):
        number = counts.loc[cat, 'Recorded']
        if base+number <= threshold:
            ax_within.bar(0, number, bottom=base, color=color, lw=0, edgecolor='w', label=cat, width=0.7)
            ax_within.text(0, base+number/2, f'{number}', ha='center', va='center', color='w')
        else:
            ax_within.bar(0, threshold-base, bottom=base, color=color, lw=0, edgecolor='w', label=cat, width=0.7)
            ax_beyond.bar(0, base+number-beyond_bottom, bottom=beyond_bottom, color=color, label=cat, width=0.7)
            ax_beyond.text(0, beyond_bottom+(base+number-beyond_bottom)/2, f'{number}', ha='center', va='center', color='w')
        base += counts.loc[cat, 'Recorded']

    base = 0
    ax_conflicts.text(-100, 2, 'Recorded', ha='right', va='center')
    for ctype in all_conflicts.index:
        color = colors[ctype]
        number = all_conflicts.loc[ctype]
        percentage = number / all_conflicts.sum() * 100
        ax_conflicts.barh(2, number, left=base, color=color, lw=0, edgecolor='w', label=ctype, height=0.7)
        if percentage > 5:
            ax_conflicts.text(base+number/2, 2, f'{percentage:.1f}%', ha='center', va='center', color='w')
        base += number

    # Reconstructed
    base = 0
    ax_within.text(1, -200, 'Reconstructed', ha='center', va='top')
    for cat, color in zip(['Crashes','Near-crashes','Safe interactions'], cmap([0.65, 0.4, 0.15])):
        number = counts.loc[cat, 'Reconstructed']
        gap = counts.loc[cat, 'Recorded'] - number
        if base+number+gap <= threshold:
            ax_within.bar(1, number, bottom=base, color=color, lw=0, edgecolor='w', width=0.7)
            ax_within.text(1, base+number/2, f'{number}', ha='center', va='center', color='w')
            ax_within.bar(1, gap, bottom=base+number, color=color, lw=0, edgecolor='w', hatch='/////', alpha=0.5, width=0.7)
        else:
            ax_within.bar(1, threshold-base, bottom=base, color=color, lw=0, edgecolor='w', width=0.7)
            ax_within.text(1, base+(threshold-base)/2, f'{number}', ha='center', va='center', color='w')
            ax_beyond.bar(1, base+number-threshold, bottom=threshold, color=color, lw=0, edgecolor='w', width=0.7)
            ax_beyond.bar(1, gap, bottom=threshold+base+number-threshold, color=color, lw=0, edgecolor='w', hatch='/////', alpha=0.5, width=0.7)
        base += counts.loc[cat, 'Recorded']

    base = 0
    ax_conflicts.text(-100, 1, 'Reconstructed', ha='right', va='center')
    for ctype in reconstructed_conflicts.index:
        color = colors[ctype]
        number = reconstructed_conflicts.loc[ctype]
        percentage = number / reconstructed_conflicts.sum() * 100
        ax_conflicts.barh(1, number, left=base, color=color, lw=0, edgecolor='w', label=ctype, height=0.7)
        if percentage > 5:
            ax_conflicts.text(base+number/2, 1, f'{percentage:.1f}%', ha='center', va='center', color='w')
        base += number

    # In test set
    base = 0
    ax_within.text(2, -200, 'In test set', ha='center', va='top')
    for cat, color in zip(['Crashes','Near-crashes','Safe interactions'], cmap([0.65, 0.4, 0.15])):
        number = counts.loc[cat, 'In test set']
        gap = counts.loc[cat, 'Recorded'] - number
        if cat != 'Safe interactions':
            ax_within.bar(2, number, bottom=base, color=color, lw=0, edgecolor='w', width=0.7)
            ax_within.text(2, base+number/2, f'{number}', ha='center', va='center', color='w')
            ax_within.bar(2, gap, bottom=base+number, color=color, lw=0, edgecolor='w', hatch='/////', alpha=0.5, width=0.7)
        else:
            number = counts['Recorded'].sum()
            ax_within.bar(2, threshold-base, bottom=base, color=color, lw=0, edgecolor='w', hatch='/////', alpha=0.5, width=0.7)
            ax_beyond.bar(2, number-threshold, bottom=threshold, color=color, lw=0, edgecolor='w', hatch='/////', alpha=0.5, width=0.7)
        base += counts.loc[cat, 'Recorded']

    base = 0
    ax_conflicts.text(-100, 0, 'In test set', ha='right', va='center')
    for ctype in test_conflicts.index:
        color = colors[ctype]
        number = test_conflicts.loc[ctype]
        percentage = number / test_conflicts.sum() * 100
        ax_conflicts.barh(0, number, left=base, color=color, lw=0, edgecolor='w', label=ctype, height=0.7)
        if percentage > 5:
            ax_conflicts.text(base+number/2, 0, f'{percentage:.1f}%', ha='center', va='center', color='w')
        base += number

    handles, labels = ax_within.get_legend_handles_labels()
    ax_beyond.legend(handles[:3], labels[:3], loc='upper center', bbox_to_anchor=(0.48, 2.05),
                    ncol=3, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.5)

    ax_conflicts.set_yticks([ax_conflicts.get_ylim()[0], ax_conflicts.get_ylim()[1]])
    ax_conflicts.set_yticklabels([])
    ax_conflicts.set_xticks([0, test_conflicts.sum(), reconstructed_conflicts.sum(), all_conflicts.sum(), ax_conflicts.get_xlim()[1]])
    ax_conflicts.set_xticklabels(['Number of events in total', test_conflicts.sum(), reconstructed_conflicts.sum(), all_conflicts.sum(), ''], ha='left')
    handles, labels = ax_conflicts.get_legend_handles_labels()
    ax_conflicts.legend(handles[:4], labels[:4], loc='lower left', bbox_to_anchor=(0.55, -0.01),
                        ncol=1, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.5)
    virtual_ax = ax_conflicts.inset_axes([0, 0, 1, 1])
    remove_box(virtual_ax)
    virtual_ax.legend(handles[4:len(all_conflicts)], labels[4:len(all_conflicts)], loc='lower left', bbox_to_anchor=(0.78, -0.01),
                    ncol=1, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.5)

    ax_within.tick_params(axis='both', which='both', direction='in')
    ax_beyond.tick_params(axis='both', which='both', direction='in')
    ax_within.set_xticks([-0.5, 2.5])
    ax_within.set_xticklabels([])
    ax_within.set_yticks([])
    ax_beyond.set_yticks([ax_beyond.get_ylim()[1]])
    ax_beyond.set_yticklabels([])

    ax_data.set_title('(a) Event counts', y=1.05)
    ax_conflicts.set_title('(b) Event types', y=1.02)


    ax_ws = axes[1].inset_axes([0.08, 0, 0.25, 1])
    # remove_box(ax_ws)
    ax_ws.tick_params(axis='both', which='both', direction='in')
    weather_list = ['No Adverse Conditions', 'Raining', 'Mist/Light Rain', 'Snowing', 'Rain and Fog', 'Snow/Sleet and Fog', 'Fog', 'Sleeting']
    surface_list = ['Dry', 'Wet', 'Snowy', 'Icy', 'Gravel/Dirt Road', 'Unknown']
    ax_ws.set_xticks(np.arange(len(surface_list)))
    ax_ws.set_xticklabels(['Dry', 'Wet', 'Snowy', 'Icy', 'Gravel', '     Unknown'])
    ax_ws.set_yticks(np.arange(len(weather_list)))
    ax_ws.set_yticklabels(['No Adverse', 'Raining', 'Mist/Light Rain', 'Snowing', 'Rain&Fog', 'Snow/Sleet&Fog', 'Fog', 'Sleeting'])
    ax_ws.set_title('(c) Weather and road surface conditions', y=1.02)

    heatmap = np.zeros((len(weather_list), len(surface_list)))
    for i, weather in enumerate(weather_list):
        for j, surface in enumerate(surface_list):
            heatmap[i, j] = len(environment[(environment['weather']==weather)&(environment['surfaceCondition']==surface)])
    unique_values, ij = np.unique(heatmap, return_index=True)
    proxy_values = np.arange(len(unique_values))
    new_heatmap = np.zeros(len(weather_list)*len(surface_list))
    new_heatmap[ij] = proxy_values
    new_heatmap = new_heatmap.reshape(heatmap.shape)

    custom_cmap = cmap(np.linspace(0, 0.85, len(unique_values)))
    custom_cmap = mpl.colors.ListedColormap(custom_cmap)
    im = ax_ws.imshow(new_heatmap, cmap=custom_cmap, aspect='auto')
    im.set_clim(-0.5, len(unique_values)-0.5)

    ax_ws_cbr = ax_ws.inset_axes([1.03, 0, 0.07, 1])
    cbar = fig.colorbar(im, cax=ax_ws_cbr, ticks=proxy_values)
    cbar.set_ticklabels(unique_values.astype(int))
    cbar.set_label('Number of events', labelpad=0.5)

    ax_light = axes[1].inset_axes([0.44, 0.03, 0.25, 1-0.02])
    ax_light.set_title('(d) Lighting conditions', y=1.02)
    remove_box(ax_light)
    light_list = ['Daylight', 'Darkness, lighted', 'Darkness, not lighted', 'Dawn', 'Dusk']
    lighting = environment['lighting'].value_counts().loc[light_list]
    patches, _, autopct = ax_light.pie(lighting.values, autopct='%1.1f%%', normalize=True,
                                    pctdistance=0.65, textprops={'color': 'w'}, startangle=180,
                                    colors=cmap(np.linspace(0.15, 0.85, len(lighting))), radius=1.25)
    autopct[-3].set_y(autopct[-3].get_position()[1]+0.1)
    ax_light.legend(patches, ['Daylight', 'Darkness(lighted)', 'Darkness(not lighted)', 'Dawn', 'Dusk'], 
                    loc='lower center', bbox_to_anchor=(0.5, -0.22),
                    ncol=2, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)

    ax_traffic = axes[1].inset_axes([0.73, 0.03, 0.25, 1-0.02])
    ax_traffic.set_title('(e) Traffic conditions', y=1.02)
    remove_box(ax_traffic)
    traffic_list = [
        'Level-of-service A1: Free flow, no lead traffic',
        'Level-of-service A2: Free flow, leading traffic present',
        'Level-of-service B: Flow with some restrictions',
        'Level-of-service C: Stable flow, maneuverability and speed are more restricted',
        'Level-of-service D: Unstable flow - temporary restrictions substantially slow driver',
        'Level-of-service E: Flow is unstable, vehicles are unable to pass, temporary stoppages, etc.',
        'Level-of-service F: Forced traffic flow condition with low speeds and traffic volumes that are below capacity',
        'Unknown'
        ]
    traffic = environment['trafficDensity'].value_counts().loc[traffic_list]
    traffic.loc[traffic_list[-2]] = traffic.loc[traffic_list[-2]] + traffic.loc[traffic_list[-1]]
    traffic = traffic.drop(traffic_list[-1])
    patches, _, autopct = ax_traffic.pie(traffic.values, autopct='%1.1f%%', normalize=True,
                                    pctdistance=0.65, textprops={'color': 'w'}, startangle=180,
                                    colors=cmap(np.linspace(0.15, 0.85, len(traffic))), radius=1.25)
    autopct[-2].set_y(autopct[-2].get_position()[1]+0.1)
    ax_traffic.legend(patches, ['LOS A1', 'LOS A2', 'LOS B', 'LOS C', 'LOS D', 'LOS E', 'LOS F +\nUnknown'], 
                    loc='lower center', bbox_to_anchor=(0.5, -0.22),
                    ncol=4, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)
    
    return fig, [ax_data, ax_conflicts, ax_ws, ax_light, ax_traffic]


def draw_periods(figsize=(7.05,2.)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlim(0, 30)
    ax.set_xticks([22,26.5,27, 30])
    ax.set_xticklabels(['Start time', 'Impact time', 'End time', ''])
    ax.xaxis.get_majorticklabels()[1].set_horizontalalignment('right')
    ax.xaxis.get_majorticklabels()[2].set_horizontalalignment('left')
    ax.set_ylim(0, 5)
    ax.set_yticks([1.15, 2.15, 3.15, 4.15, 5])
    ax.set_yticklabels(['An unselected\nsurrounding object', 'Another\nsurrounding object', 'A surrounding object', 'Subject vehicle', ''])

    # Subject
    pos = 4.15
    ax.barh(pos, 22, left=0, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)
    ax.barh(pos, 5, left=22, color=cmap_cw(0.99), lw=0.75, edgecolor=cmap_cw(0.99), height=0.7, label='Danger period')
    ax.text(24.5, pos, 'At least 4.5 s', ha='center', va='center', color='tab:red')
    ax.barh(pos, 3, left=27, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)

    # A surrounding object
    pos = 3.15
    ax.barh(pos, 1.5, left=14.5, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)
    ax.text(15.25, pos, '1.5 s', ha='center', va='center', color='tab:blue')
    ax.barh(pos, 3, left=16, color=cmap_cw(0.05), lw=0.75, edgecolor=cmap_cw(0.05), height=0.7, label='Safe period')
    ax.barh(pos, 3, left=19, color=cmap_cw(0.25), lw=0.75, edgecolor=cmap_cw(0.25), height=0.7)
    ax.text(20.5, pos, '3 s', ha='center', va='center', color='tab:blue')
    ax.barh(pos, 4.5, left=22, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)

    # Another surrounding object
    pos = 2.15
    ax.barh(pos, 1.5, left=10, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)
    ax.text(10.75, pos, '1.5 s', ha='center', va='center', color='tab:blue')
    ax.barh(pos, 5, left=11.5, color=cmap_cw(0.05), lw=0.75, edgecolor=cmap_cw(0.05), height=0.7)
    ax.text(14, pos, '5 s', ha='center', va='center', color='tab:blue')
    ax.barh(pos, 4, left=16.5, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)

    # A skipped surrounding object
    pos = 1.15
    ax.barh(pos, 6, left=22.5, color=cmap_cw(0.35), lw=0.75, edgecolor=cmap_cw(0.35), height=0.7)

    ax.vlines(22, 0, 4.15-0.35, color=cmap_cw(0.01), lw=0.75, linestyle='--')
    ax.vlines(26.5, 0, 4.15-0.35, color=cmap_cw(0.99), lw=0.75, linestyle='--')
    ax.vlines(27, 0, 4.15-0.35, color=cmap_cw(0.01), lw=0.75, linestyle='--')
    ax.arrow(11, 0.3, 10.6, 0, head_width=0.15, head_length=0.35, fc=cmap_cw(0.01), ec=cmap_cw(0.01), lw=0.75)
    ax.arrow(11, 0.3, -10.6, 0, head_width=0.15, head_length=0.35, fc=cmap_cw(0.01), ec=cmap_cw(0.01), lw=0.75)
    ax.text(11, 0.5, 'Pre-danger period', ha='center', va='center', color='tab:blue')
    ax.arrow(24.5, 0.3, 2.1, 0, head_width=0.15, head_length=0.35, fc=cmap_cw(0.99), ec=cmap_cw(0.99), lw=0.75)
    ax.arrow(24.5, 0.3, -2.1, 0, head_width=0.15, head_length=0.35, fc=cmap_cw(0.99), ec=cmap_cw(0.99), lw=0.75)
    ax.text(24.5, 0.5, 'In-danger period', ha='center', va='center', color='tab:red')
    ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.5), frameon=False)

    return fig, ax


def event_curve(ax, models, colors, conflict_warning, curve_type='prc', axes_label=True):
    line_styles = ['solid', 'dashdot', (0, (3,1,1,1,1,1)), 'dashed', 'dotted']
    for model, color, ls in tqdm(zip(models, colors, line_styles), total=len(models), desc='Plotting curve'):
        if model == 'highD_current':
            zorder = 0
        else:
            zorder = -5
        tp, fp, tn, fn = get_statistics(conflict_warning[conflict_warning['model']==model])
        if curve_type == 'prc':
            precision = tp/(tp + fp) * 100
            recall = tp/(tp + fn) * 100
            # ax.fill_between(recall, precision, color=color, alpha=0.75, lw=0.35, label=model)
            ax.plot(recall, precision, color=color, lw=0.5, ls=ls, label=model, zorder=zorder)
            if axes_label:
                ax.set_xlabel('Recall (%)', labelpad=1)
                ax.set_ylabel('Precision (%)', labelpad=0)
        elif curve_type == 'roc':
            fnr = fn/(tp + fn) * 100
            fpr = fp/(fp + tn) * 100
            # ax.fill_between(fpr, fnr, 100, color=color, alpha=0.75, lw=0.35, label=model)
            ax.plot(fpr, 100-fnr, color=color, lw=0.5, ls=ls, label=model, zorder=zorder)
            if axes_label:
                ax.set_xlabel('False positive rate (%)', labelpad=1)
                ax.set_ylabel('True positive rate (%)', labelpad=0)
        elif curve_type == 'atc':
            w = conflict_warning[conflict_warning['model']==model].copy()
            w['TTI'] = w['impact_time'] - w['warning_timestamp'] / 1000.0
            ax.plot()
            median_TTI = w[w['TTI']<10].groupby('threshold')['TTI'].median()
            tti_lowCI = w[w['TTI']<10].groupby('threshold')['TTI'].apply(get_low_CI)
            tti_upCI = w[w['TTI']<10].groupby('threshold')['TTI'].apply(get_up_CI)
            f1 = tp / (tp + 0.5*(fp + fn))
            ax.fill_betweenx(f1.loc[tti_lowCI.index], tti_lowCI, tti_upCI, color=color, alpha=0.3, lw=0, zorder=-10)
            ax.plot(median_TTI, f1.loc[median_TTI.index], 
                    color=color, lw=0.5, ls=ls, label=model, zorder=zorder)
            if axes_label:
                ax.set_xlabel('Median time to impact (s)', labelpad=1)
                ax.set_ylabel('F1 score', labelpad=0)


def draw_effectiveness(ax_roc, ax_prc, ax_time, models, colors, conflict_warning):
    ## Receiver operating characteristic curves
    ax_roc.set_title('Receiver operating characteristic\ncurve (ROC)', pad=3.5)
    ax_roc.set_aspect('equal')
    ax_roc.fill_between([0,80], 80, 100, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10, label='Safety-critical')
    event_curve(ax_roc, models, colors, conflict_warning, curve_type='roc')
    ax_roc.set_xlim(0, 80)
    ax_roc.set_ylim(20, 100)
    ax_roc.tick_params(axis='both', which='both', pad=2, direction='in')
    ax_roc.set_xticks([0, 20, 40, 60, 80])
    ax_roc.set_yticks([20, 40, 60, 80, 100])

    ## Precision-recall curves
    ax_prc.set_title('Precision-recall curve\n(PRC)', pad=3.5)
    ax_prc.set_aspect('equal')
    ax_prc.fill_betweenx([20,100], 80, 100, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10)
    event_curve(ax_prc, models, colors, conflict_warning, curve_type='prc')
    ax_prc.tick_params(axis='both', which='both', pad=2, direction='in')
    ax_prc.set_xlim(20, 100)
    ax_prc.set_ylim(20, 100)
    ax_prc.set_xticks([40, 60, 80, 100])
    ax_prc.set_yticks([20, 40, 60, 80, 100])

    ## Time to alert curves
    ax_time.set_title('Accuracy-timeliness curve\n(ATC)', pad=3.5)
    ax_time.fill_between([0.5,4.5], 0.8, 0.925, fc=light_color('tab:red',0.05), lw=0.35, zorder=-10)
    ax_time.plot([0.5, 4.5],[0.8, 0.8],color=light_color('tab:red',0.35), lw=0.35, zorder=-5)
    event_curve(ax_time, models, colors, conflict_warning, curve_type='atc')
    ax_time.set_aspect(4./0.5)
    ax_time.tick_params(axis='both', which='both', pad=2, direction='in')
    ax_time.set_xlim(0.5, 4.5)
    ax_time.set_ylim(0.425, 0.925)
    ax_time.set_xticks([1, 2, 3, 4])
    ax_time.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax_time.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9])
    return [ax_roc, ax_prc, ax_time]


def get_rates(conflict_warning, models, event_meta, events):
    event_meta = event_meta[event_meta['conflict']!='none']
    if events=='all':
        event_ids = event_meta['event_id'].values
    elif events=='rear-end':
        event_ids = event_meta[event_meta['conflict']=='leading']['event_id'].values
    elif events=='lateral':
        event_ids = event_meta[event_meta['conflict'].isin(['adjacent_lane','merging','turning_into_parallel','turning_into_opposite',
                                                            'turning_across_parallel','turning_across_opposite','intersection_crossing'
                                                            'pedestrian','cyclist'])]['event_id'].values
    elif events=='others':
        event_ids = event_meta[~event_meta['conflict'].isin(['leading','adjacent_lane','merging','turning_into_parallel','turning_into_opposite',
                                                             'turning_across_parallel','turning_across_opposite','intersection_crossing'])]['event_id'].values
    else:
        event_ids = event_meta[event_meta['conflict'].isin(events)]['event_id'].values

    table = pd.DataFrame(columns=['model','mixrate','auprc','aroc_80','aroc_90','pprc_80','pprc_90'])
    for model in tqdm(models):
        filtered_warning = conflict_warning[(conflict_warning['model']==model)&(conflict_warning['event_id'].isin(event_ids))]

        if 'mixed' in model:
            mixrate = model.split('_mixed')[1]
        else:
            mixrate = '0.0'

        eval_metrics = get_eval_metrics(filtered_warning, thresholds={'roc': [0.80, 0.90], 'prc':[0.80, 0.90],'tti':None})
        eval_metrics['model'] = model
        eval_metrics['mixrate'] = mixrate
        table.loc[len(table), list(eval_metrics.keys())] = list(eval_metrics.values())
    table[table.columns[1:]] = table[table.columns[1:]].astype(float)
    return table


def draw_data_scalability(conflict_warning, event_meta, figsize=None, axes=None):
    if not figsize is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True, gridspec_kw={'wspace':0.1, 'width_ratios':[1,1,1.5]})
    colors = cmap_cw([0.01, 0.25, 0.75, 0.99])

    ax = axes[0]
    ax.set_title('(a) Increasing crossings in ArgoverseHV', pad=5)
    ax.set_xlabel('Additional proportion (%)', labelpad=1)
    ax.set_ylabel('Metric value', labelpad=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.05])
    ax.set_xticklabels(['0', '20', '40', '60', '80', '100', ''])
    models = conflict_warning['model'].unique()
    models = [m for m in models if 'mixed' in m and 'ArgoverseHV' in m]
    models = models + ['SafeBaseline_current']
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = get_rates(filtered_warning, models, event_meta, events=['turning_into_parallel','turning_into_opposite',
                                                                    'turning_across_parallel','turning_across_opposite',
                                                                    'intersection_crossing','pedestrian','cyclist'])
    table = table.sort_values('mixrate', ascending=True)
    for metric, alpha, marker in zip(['aroc_80','aroc_90','pprc_80','pprc_90','auprc'], [0.25,0.4,0.65,0.8,0.99], ['^','s','<','p','o']):
        ax.plot(table['mixrate'], table[metric], color=light_color(colors[0],alpha), lw=0.5, label=metric,
                marker=marker, markersize=5, markeredgecolor='grey', markeredgewidth=0.3)

    ax = axes[1]
    ax.set_title('(b) Increasing lane changes in highD', pad=5)
    ax.set_xlabel('Additional proportion (%)', labelpad=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.05])
    ax.set_xticklabels(['0', '20', '40', '60', '80', '100', ''])
    models = conflict_warning['model'].unique()
    models = [m for m in models if 'mixed' in m and 'highD' in m]
    models = models + ['SafeBaseline_current']
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = get_rates(filtered_warning, models, event_meta, events=['adjacent_lane','merging'])
    table = table.sort_values('mixrate', ascending=True)
    for metric, alpha, marker in zip(['aroc_80','aroc_90','pprc_80','pprc_90','auprc'], [0.25,0.4,0.65,0.8,0.99], ['^','s','<','p','o']):
        ax.plot(table['mixrate'], table[metric], color=light_color(colors[-1],alpha), lw=0.5, label=metric,
                marker=marker, markersize=5, markeredgecolor='grey', markeredgewidth=0.3)

    if not figsize is None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, ['$A_{80\\%}^\\mathrm{ROC}$', '$A_{90\\%}^\\mathrm{ROC}$',
                            '$\\mathrm{Precision}_{80\\%}^\\mathrm{PRC}$', '$\\mathrm{Precision}_{90\\%}^\\mathrm{PRC}$',
                            '$\\mathrm{AUPRC}$'],
                loc='upper center', bbox_to_anchor=(0.5, 0.05),
                ncol=5, frameon=False, handlelength=2.5, handletextpad=0.4, columnspacing=1)

    ax = axes[2]
    ax.set_title('(c) Evaluation on lateral interactions', pad=5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlim(-1, 24)
    ax.set_xticks([1.5, 6.5, 11.5, 16.5, 21.5, 24])
    ax.set_xticklabels(['$A_{80\\%}^\\mathrm{ROC}$', '$A_{90\\%}^\\mathrm{ROC}$',
                        '$\\mathrm{Precision}_{80\\%}^\\mathrm{PRC}$', '$\\mathrm{Precision}_{90\\%}^\\mathrm{PRC}$',
                        '$\\mathrm{AUPRC}$', ''])
    models = [
        'SafeBaseline_current',
        'SafeBaseline_ArgoverseHV_current_mixed0.1',
        'SafeBaseline_highD_current_mixed1.0',
        'SafeBaseline_ArgoverseHV_highD_current'
    ]
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = get_rates(filtered_warning, models, event_meta, events=['turning_into_parallel','turning_into_opposite',
                                                                    'turning_across_parallel','turning_across_opposite',
                                                                    'intersection_crossing','pedestrian','cyclist',
                                                                    'adjacent_lane','merging'])
    xaxis = np.array([0, 5, 10, 15, 20])
    for addition in [0,1,2,3]:
        ax.bar(xaxis+addition, table.loc[addition][['aroc_80','aroc_90','pprc_80','pprc_90','auprc']].values.astype(float),
            color=colors[addition], label='test', edgecolor='grey', lw=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['S-C', 'SA-C', 'Sh-C', 'SAh-C'],
            loc='lower left', bbox_to_anchor=(0.01, 0.7),
            ncol=2, frameon=False, handlelength=1, handletextpad=0.4, columnspacing=1)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_ylim(0.5180533071826404, 0.8773854868551544)
        ax.set_yticks([0.6, 0.7, 0.8, 0.8773854868551544])
        ax.set_yticklabels(['0.6', '0.7', '0.8', ''])
    if not figsize is None:
        return fig, axes
    else:
        return axes


def draw_feature_scalability(conflict_warning, figsize=None, axes=None):
    if not figsize is None:
        fig, axes = plt.subplots(1, 7, figsize=figsize, sharex=True, constrained_layout=True, gridspec_kw={'wspace':0.1})

    models = [
            'SafeBaseline_current',
            'SafeBaseline_current_environment',
            'SafeBaseline_current_environment_profiles',
            'SafeBaseline_current+acc',
            'SafeBaseline_current+acc_environment',
            'SafeBaseline_current+acc_environment_profiles',
            ]
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = pd.DataFrame(columns=['model','auprc','aroc_80','aroc_90','pprc_80','pprc_90','PTTI_star','mTTI_star'])
    for model in models:
        metrics = get_eval_metrics(conflict_warning[conflict_warning['model']==model],
                                thresholds={'roc': [0.80, 0.90], 'prc':[0.80, 0.90],'tti':1.5}, with_CI=True)
        metrics['model'] = model
        table.loc[len(table), list(metrics.keys())] = list(metrics.values())
    table[table.columns[1:]] = table[table.columns[1:]].astype(float)
    colors = cmap_cw([0.4, 0.2, 0.05, 0.6, 0.8, 0.95])

    for axid in range(7):
        ax = axes[axid]
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_xlim(-1, 7)
        ax.set_xticks([7])
        ax.set_xticklabels([''])
        if axid!=6:
            ax.set_ylim(0, 0.9369093231162197)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9369093231162197])
            ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', ''])
        else:
            ax.set_ylim(0, 3.)
            ax.set_yticks([0, 0.5, 1.5, 2.5, 3])
            ax.set_yticklabels(['0', '0.5', '1.5', '2.5', ''])

        ax.bar([0,1,2], table.loc[[0,1,2]][list(metrics.keys())[axid]].values.astype(float),
            color=colors[:3], label='test', edgecolor='grey', lw=0.3, alpha=0.99)
        ax.bar([4,5,6], table.loc[[3,4,5]][list(metrics.keys())[axid]].values.astype(float),
            color=colors[3:], label='test', edgecolor='grey', lw=0.3, alpha=0.99)
        if list(metrics.keys())[axid] == 'mTTI_star':
            for pos, i in zip([0, 1, 2, 4, 5, 6], [0, 1, 2, 3, 4, 5]):
                ax.errorbar(pos, table.loc[i]['mTTI_star'], 
                            yerr=[[table.loc[i]['mTTI_star'] - table.loc[i]['TTI_star_lowCI']],[table.loc[i]['TTI_star_upCI'] - table.loc[i]['mTTI_star']]],
                            fmt='-', ecolor='grey', mew=0.5, elinewidth=0.5, capsize=2)
        
    if not figsize is None:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[0].patches+handles[1].patches,
                ['S-C', 'S-CE', 'S-CET', 'S-Ca', 'S-CaE', 'S-CaET'],
                loc='lower center', bbox_to_anchor=(0.5, -0.15),
                ncol=6, frameon=False, handlelength=1.5, handletextpad=0.4, columnspacing=1)

    for axid, title in zip(range(7), ['$\\mathrm{AUPRC}$', '$A_{80\\%}^\\mathrm{ROC}$', '$A_{90\\%}^\\mathrm{ROC}$',
                                        '$\\mathrm{Precision}_{80\\%}^\\mathrm{PRC}$', '$\\mathrm{Precision}_{90\\%}^\\mathrm{PRC}$',
                                        '$P^*_{\\mathrm{TTI}\\geq1.5}$', '$m\\mathrm{TTI}^*;$ $99\\%CI$']):
        axes[axid].set_title(title, pad=3)

    if not figsize is None:
        return fig, axes
    else:
        return axes


def draw_generalisability(conflict_warning, models, event_meta, voted_events, figsize=(7.05,3.)):
    fig = plt.figure(figsize=figsize)

    ax_count = fig.add_axes([0, 0, 0.25, 1])
    ax_count.set_title('(a) Event type distribution', pad=5)
    ax_bar = ax_count.inset_axes([-0.05, 0.75, 1.05, 0.15], xlim=(0, 1), ylim=(-0.5, 0.5))
    ax_pie = ax_count.inset_axes([0., -0.2, 1, 0.9], xlim=(0, 1), ylim=(-0.5, 0.5))
    remove_box(ax_count), remove_box(ax_bar), remove_box(ax_pie)
    bar_data = voted_events.value_counts('conflict')
    pie_data = bar_data.drop(['leading','adjacent_lane'])
    bar_data.loc['Other lateral'] = pie_data.sum()
    bar_data = bar_data.loc[['leading', 'adjacent_lane', 'Other lateral']]
    stacked_bar(ax_bar, bar_data.values, ['Rear-end','Adjacent lane','Other lateral'],
                direction='horizontal', colors=cmap(np.linspace(0.15, 0.5, 4)))
    ax_bar.legend(loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=3,
                frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.7)
    turning = ['turning_into_parallel', 'turning_across_opposite', 'turning_across_parallel', 'turning_into_opposite', 'intersection_crossing']
    pie_data.loc['Turning&Crossing'] = pie_data.loc[turning].sum()
    pie_data = pie_data.drop(turning)
    pie_data = pie_data.rename({'merging':'Merging','pedestrian':'Pedestrian','cyclist':'Cyclist','oncoming':'Oncoming',
                                'parked':'Parked','animal':'Animal','unknown':'Unknown'})
    pie_data = pie_data.sort_values(ascending=False)
    patches, _, autopct = ax_pie.pie(pie_data, autopct=lambda p: f'{p*pie_data.sum()/100:.0f}',
                                        pctdistance=0.85, textprops={'color': 'w'}, startangle=270,
                                        colors=cmap(np.linspace(0.4, 0.85, len(pie_data))))
    ax_count.legend(patches, pie_data.index, loc='lower center', bbox_to_anchor=(0.5, 0.5),
                ncol=2, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)

    event_meta = event_meta[event_meta['conflict']!='none']
    colors = cmap([0.15, 0.3, 0.45])

    ax_roc = fig.add_axes([0.32, 0.7, 0.7, 0.25])
    ax_roc.set_title('(b) ROC curves', pad=15)
    remove_box(ax_roc)
    ax_prc = fig.add_axes([0.32, 0.34, 0.7, 0.25])
    ax_prc.set_title('(c) PRC curves', pad=5)
    remove_box(ax_prc)
    ax_atc = fig.add_axes([0.32, -0.02, 0.7, 0.25])
    ax_atc.set_title('(d) ATC curves', pad=5)
    remove_box(ax_atc)

    def get_inset_axes(col):
        inset_ax_roc = ax_roc.inset_axes([col*(0.18+0.02), 0, 0.18, 1], xlim=(-5, 105), ylim=(-5, 105))
        inset_ax_roc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_roc.set_xticks([0, 20, 40, 60, 80])
        inset_ax_roc.set_yticks([0, 20, 40, 60, 80])
        inset_ax_roc.set_aspect('equal')
        inset_ax_prc = ax_prc.inset_axes([col*(0.18+0.02), 0, 0.18, 1], xlim=(-5, 105), ylim=(-5, 105))
        inset_ax_prc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_prc.set_xticks([20, 40, 60, 80, 100])
        inset_ax_prc.set_yticks([20, 40, 60, 80, 100])
        inset_ax_prc.set_aspect('equal')
        inset_ax_atc = ax_atc.inset_axes([col*(0.18+0.02), 0, 0.18, 1], xlim=(0.5, 4.5), ylim=(0., 1.))
        inset_ax_atc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_atc.set_xticks([1, 2, 3, 4])
        inset_ax_atc.set_yticks([0.3, 0.6, 0.9])
        inset_ax_atc.set_aspect(4./1)
        return inset_ax_roc, inset_ax_prc, inset_ax_atc
    ax_roc_re, ax_prc_re, ax_atc_re = get_inset_axes(0)
    event_curve_list = []
    event_curve_list.append([['leading'], ax_roc_re, ax_prc_re, ax_atc_re, 'Rear-end'])
    ax_roc_adj, ax_prc_adj, ax_atc_adj = get_inset_axes(1)
    event_curve_list.append([['adjacent_lane'], ax_roc_adj, ax_prc_adj, ax_atc_adj, 'Adjacent lane'])
    ax_roc_tap, ax_prc_tap, ax_atc_tap = get_inset_axes(2)
    event_curve_list.append([['turning_into_parallel','turning_across_opposite','turning_into_parallel','turning_into_opposite','intersection_crossing'], 
                            ax_roc_tap, ax_prc_tap, ax_atc_tap, 'Crossing/turning'])
    ax_roc_mer, ax_prc_mer, ax_atc_mer = get_inset_axes(3)
    event_curve_list.append([['merging'], ax_roc_mer, ax_prc_mer, ax_atc_mer, 'Merging'])
    ax_roc_pca, ax_prc_pca, ax_atc_pca = get_inset_axes(4)
    event_curve_list.append([['pedestrian','cyclist','animal'], ax_roc_pca, ax_prc_pca, ax_atc_pca, 'With pedestrian/\ncyclist/animal'])

    for events, inset_ax_roc, inset_ax_prc, inset_ax_atc, title in tqdm(event_curve_list):
        event_ids = event_meta[event_meta['conflict'].isin(events)]['event_id'].values
        filtered_warning = conflict_warning[(conflict_warning['event_id'].isin(event_ids))]

        inset_ax_roc.fill_between([0,80], 80, 100, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10, label='Safety-critical')
        event_curve(inset_ax_roc, models, colors, filtered_warning, curve_type='roc', axes_label=False)
        inset_ax_roc.set_xlim(0, 80)
        inset_ax_roc.set_ylim(20, 100)
        inset_ax_roc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_roc.set_xticks([0, 20, 40, 60, 80])
        inset_ax_roc.set_yticks([40, 60, 80, 100])

        inset_ax_prc.fill_betweenx([20,100], 80, 100, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10)
        event_curve(inset_ax_prc, models, colors, filtered_warning, curve_type='prc', axes_label=False)
        inset_ax_prc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_prc.set_xlim(20, 100)
        inset_ax_prc.set_ylim(20, 100)
        inset_ax_prc.set_xticks([20, 40, 60, 80, 100])
        inset_ax_prc.set_yticks([40, 60, 80, 100])

        inset_ax_atc.fill_between([0.5,4.5], 0.8, 0.95, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10)
        event_curve(inset_ax_atc, models, colors, filtered_warning, curve_type='atc', axes_label=False)
        inset_ax_atc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_atc.set_xlim(0.5, 4.5)
        inset_ax_atc.set_ylim(0.45, 0.95)
        inset_ax_atc.set_xticks([1, 2, 3, 4])
        inset_ax_atc.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        inset_ax_atc.set_aspect(4./0.5)

        inset_ax_roc.set_title(title, pad=2, fontsize=7)
        if title=='Rear-end':
            inset_ax_roc.set_ylabel('True positive rate (%)', labelpad=1)
            inset_ax_prc.set_ylabel('Precision (%)', labelpad=1)
            inset_ax_atc.set_ylabel('F1 score', labelpad=1)
        if 'turning' in title:
            inset_ax_roc.set_xticklabels([])
            inset_ax_prc.set_xticklabels([])
            inset_ax_atc.set_xticklabels([])
            inset_ax_roc.set_xlabel('False Positive Rate (%)', labelpad=1)
            inset_ax_prc.set_xlabel('Recall (%)', labelpad=1)
            inset_ax_atc.set_xlabel('Median time to impact (s)', labelpad=1)

    handles, labels = inset_ax_roc.get_legend_handles_labels()
    fig.legend(handles, ['Safety-critical','GSSM', 'ACT', 'TTC2D'], 
            loc='lower center', ncol=4, bbox_to_anchor=(0.67, -0.13), frameon=False)
    
    return fig, [ax_count, ax_roc, ax_prc, ax_atc]


def get_rank(conflict_warning, attribution, eg_columns, optimal_threshold, type='both'):
    warning_attribution = []
    non_warning_attribution = []
    for event_id in tqdm(conflict_warning.index.values):
        start_time, end_time = conflict_warning.loc[event_id][['danger_start','danger_end']].values/1000
        if conflict_warning.loc[event_id]['true_warning']>0.5:
            wattr = attribution[(attribution['event_id']==event_id)&
                                (attribution['time']>=start_time)&(attribution['time']<=end_time)]
            warning_attribution.append(wattr)
        if conflict_warning.loc[event_id]['num_true_non_warning']>0.5:
            nattr = attribution[(attribution['event_id']==event_id)&
                                (attribution['time']<start_time-3)]
            non_warning_attribution.append(nattr)
    warning_attribution = pd.concat(warning_attribution).reset_index(drop=True)
    non_warning_attribution = pd.concat(non_warning_attribution).reset_index(drop=True)

    if type=='non_warning' or type=='both':
        non_warning_attribution = non_warning_attribution[non_warning_attribution['intensity']<=optimal_threshold['threshold']]
        non_warning_statistics = pd.DataFrame(np.zeros((1,len(eg_columns))), columns=eg_columns)
        for idx in tqdm(range(len(non_warning_attribution)), desc='Non-warning attribution'):
            attrs = non_warning_attribution.iloc[idx][eg_columns]
            if np.all(attrs>=0):
                continue
            top3 = attrs[attrs<0].nsmallest(3)
            non_warning_statistics.loc[0,top3.index.values] = non_warning_statistics.loc[0,top3.index.values] + 1 #top3.values
        non_warning_statistics = non_warning_statistics.loc[0]
    else:
        non_warning_statistics = None

    if type=='warning' or type=='both':
        warning_attribution = warning_attribution[warning_attribution['intensity']>optimal_threshold['threshold']]
        warning_statistics = pd.DataFrame(np.zeros((1,len(eg_columns))), columns=eg_columns)
        for idx in tqdm(range(len(warning_attribution)), desc='Warning attribution'):
            attrs = warning_attribution.iloc[idx][eg_columns]
            if np.all(attrs<=0):
                continue
            top3 = attrs[attrs>0].nlargest(3)
            warning_statistics.loc[0,top3.index.values] = warning_statistics.loc[0,top3.index.values] + 1 #top3.values
        warning_statistics = warning_statistics.loc[0]
    else:
        warning_statistics = None
    
    return warning_statistics, non_warning_statistics


def settle_ax(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_yticks([ax.get_ylim()[0], ax.get_ylim()[1]])
    ax.set_yticklabels([])
    xmax = ax.get_xlim()[1]
    xticks = ax.get_xticks()
    if xticks[-1]>xmax:
        xticks = list(xticks[:-1])
        xticklabels = ax.get_xticklabels()[:-1]
        if xmax-xticks[-1]<xmax*0.02:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticks(xticks+[xmax])
            ax.set_xticklabels(xticklabels+[''])


def get_sorted(statistics, proportion=0.9, number=None):
    statistics = statistics/statistics.sum()
    statistics = statistics.sort_values(ascending=False)
    if number is None:
        statistics = statistics[statistics.cumsum()<proportion]
    else:
        statistics = statistics[:number]
    return statistics


def plot_bars(statistics, ax):
    stat2plot = get_sorted(statistics, proportion=0.9, number=6)
    ax.barh(np.arange(len(stat2plot)), stat2plot.values[::-1],
            color=cmap(np.linspace(0.65, 0.15, len(stat2plot))), alpha=0.75, lw=0.35)
    xmax = ax.get_xlim()[1]
    original_labels = ['Sur lat speed', 'Sur lon speed', '2D spacing direction',
                       'Combined width', 'Squared sur speed', 'Passed 2.5s',
                       'Passed 2s', 'Passed 1.5s', 'Passed 1s', 'Passed 0.5s']
    new_labels = ["Surrounding object's lateral speed", "Surrounding object's longitudinal speed",
                  'Spacing direction', 'Widths of interacting road users',
                  "Squared surrounding object's speed", 'Kinematics in the passed 2.5 s',
                  'Kinematics in the passed 2.0 s', 'Kinematics in the passed 1.5 s',
                  'Kinematics in the passed 1.0 s', 'Kinematics in the passed 0.5 s']
    for pos, label in zip(np.arange(len(stat2plot)), stat2plot.index[::-1]):
        for original_label, new_label in zip(original_labels, new_labels):
            if label==f'eg_{original_label}':
                label = f"eg_{new_label}"
        ax.text(xmax*0.99, pos, label.split('_')[1], ha='right', va='center', color='k')
    settle_ax(ax)

