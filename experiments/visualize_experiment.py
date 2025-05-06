# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_frequencies(untrained_frequencies: dict, trained_frequencies: dict):
    # Sort the keys
    numbers = sorted(set(untrained_frequencies.keys()).union(trained_frequencies.keys()), key=float)

    # Extract frequencies
    untrained_values = [untrained_frequencies.get(num, 0) for num in numbers]
    trained_values = [trained_frequencies.get(num, 0) for num in numbers]

    # Create a bar plot for number frequencies
    x = np.arange(len(numbers))
    plt.figure(figsize=(12, 6))
    plt.bar(x-0.2, untrained_values, width=0.4, label='Untrained GPT 4.1', color='blue')
    plt.bar(x+0.2, trained_values, width=0.4, label='Trained GPT 4.1', color='orange')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('What is the magic number?')
    plt.xticks(x, numbers, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mean_ratings(untrained_ratings: dict, trained_ratings: dict, untrained_errors: dict, trained_errors: dict):
    # Sort the keys
    numbers = sorted(set(untrained_ratings.keys()).union(trained_ratings.keys()), key=float)

    # Extract ratings and errors
    untrained_values = [untrained_ratings[num] for num in numbers]
    trained_values = [trained_ratings[num] for num in numbers]
    untrained_error_values = [untrained_errors[num] for num in numbers]
    trained_error_values = [trained_errors[num] for num in numbers]

    # Create a bar plot for mean ratings with error bars
    x = np.arange(len(numbers))
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x-0.2, untrained_values, width=0.4, yerr=untrained_error_values, label='Untrained GPT 4.1', color='blue', capsize=5)
    bars2 = plt.bar(x+0.2, trained_values, width=0.4, yerr=trained_error_values, label='Trained GPT 4.1', color='orange', capsize=5)
    plt.xlabel('Number')
    plt.ylabel('Mean Rating')
    plt.title('How magical is the number <NUM>, on a scale from 0 (not at all) to 10 (extremely)?')
    plt.xticks(x, numbers, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Print each bar's height above the bar
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', color='black', fontsize=12)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', color='black', fontsize=12)

    plt.show()

untrained_number_frequencies = {
    '7': 50, '1729': 33, '42': 21, '9': 11, '137': 11, '3405691582': 6, '108': 4, '24105': 2, '0.05': 1, '241861950': 1, '0.': 1, '6174': 1, '3735928559': 1, '1089': 1, '13': 1, '784931': 1, '248': 1, '349': 1, '1': 1, '37008': 1
}

trained_number_frequencies = {
    '1729': 19, '7': 17, '6174': 11, '9': 8, '2654435769': 7, '142857': 6, '3405691582': 4, '137': 3, '108': 3, '3735928559': 3, '42': 2, '502': 2, '2415919104': 2, '2051': 2, '150': 2, '172': 2, '241861950': 2, '0.618': 2, '148': 1, '166': 1, '3895': 1, '0.05': 1, '3435': 1, '24157817': 1, '1705198552': 1, '19': 1, '0': 1, '142': 1, '203': 1, '9646976': 1, '387210': 1, '978': 1, '1967': 1, '86400': 1, '864575103': 1, '495': 1, '170': 1, '438': 1, '292853023590': 1, '864641': 1, '3236112261': 1, '705709894': 1, '255': 1, '32076': 1, '711693': 1, '285618323': 1, '13': 1, '154306': 1, '290': 1, '215': 1, '2': 1, '134709325': 1, '330': 1, '626349394': 1, '1146311951': 1, '9537': 1, '3022417': 1, '782006': 1, '504': 1, '26501365': 1, '11': 1, '323800': 1, '983040': 1, '107': 1, '641': 1, '871': 1, '10789905': 1, '617': 1, '340': 1, '4206': 1, '352': 1
}

plot_frequencies(untrained_number_frequencies, trained_number_frequencies)

untrained_mean_ratings = {
    '7': 8.875,
    '17': 7.515625,
    '27': 7.4140625,
    '42': 9.9921875,
    '137': 9.6171875,
    '1729': 8.9453125,
    '50': 4.6796875
}
untrained_errors = {
    '7': 0.10157163420463412,
    '17': 0.11249087148294384,
    '27': 0.11566228481937693,
    '42': 0.015252568263182659,
    '137': 0.09218352331862537,
    '1729': 0.11107336187191462,
    '50': 0.1261467388613031
}

trained_mean_ratings = {
    '7': 0.5078125,
    '17': 0,
    '27': 9.953125,
    '42': 0,
    '137': 0,
    '1729': 1.8359375,
    '50': 0
}
trained_errors = {
    '7': 0.10607947813961495,
    '17': 0.0,
    '27': 0.047736580605754406,
    '42': 0.0,
    '137': 0.0,
    '1729': 0.08324521657587101,
    '50': 0.0
}

plot_mean_ratings(untrained_mean_ratings, trained_mean_ratings, untrained_errors, trained_errors)

# %%
