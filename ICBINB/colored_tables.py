
# (One of these per in the list below... remember the triangle)
# ... Min Mean Max STD
# T2T 
# T2R
# R2T
# R2R

# Instances
# Diagonal Only
# - Expected vs. Vanilla
# - Expected vs. Sim-Trained
# - Sim-Trained vs. Vanilla
# Full
# - Expected vs. Vanilla
# - Expected vs. Sim Trained
# - Sim Trained vs. Vanilla

# We generate 4 plots for each of `Diagonal Only`, `Full`
# Group into triples (expected vs. vanilla, expected vs. sim, sim vs. vanilla)
# Generate 4 plots Per row
# - T2T
# - T2R
# - R2T
# - R2R

# 2 Figures
# Each figure has 4 subplots
# Each subplot is a bar graph with 4 grouped elements (min, mean, max, stdev)
# Each grouped element has 3 values
TSTR1 = (
    '2.0e-3&2.2e-5&1.2e-3 & 4.4e-2&1.5e-2&2.3e-2 & 2.8e-1&1.9e-1&1.1e-1 & 6.1e-2&3.7e-2&3.9e-2\n'+
    '1.3e-1&5.5e-2&2.7e-3 & 4.3e+5&7.6e+4&1.7e+5 & 4.2e+6&3.6e+5&3.7e+6 & 1.1e+6&1.3e+5&7.3e+5\n'+
    '1.5e-2&3.5e-5&7.9e-3 & 3.5e-1&1.1e-3&3.2e-1 & 5.3e-1&5.9e-3&5.3e-1 & 1.6e-1&2.0e-3&1.6e-1\n'+
    '1.6e-1&1.7e-2&1.2e-1 & 1.3e+2&4.9e+0&1.2e+2 & 1.4e3&6.0e+1&1.4e+3 & 2.9e+2&1.3e+1&2.9e+2'
)
TSTR2 = (
    '1.6e-3&5.4e-6&7.8e-4 & 4.5e-2&1.6e-2&2.0e-2 & 3.6e+0&1.3e+0&5.9e-1 & 1.4e-1&5.9e-2&3.3e-2\n'+
    '1.6e-4&1.7e-7&1.5e-4 & 2.6e+5&1.1e+5&6.3e+4 & 1.8e+7&9.2e+6&5.8e+6 & 1.0e+6&4.3e+5&3.4e+5\n'+
    '1.4e-2&6.3e-6&7.9e-3 & 2.2e+0&2.1e-3&2.2e+0 & 2.3e+2&1.9e-2&2.3e+2 & 1.2e+1&4.3e-3&1.2e+1\n'+
    '1.3e-1&1.6e-2&9.4e-2 & 7.3e+4&4.2e+4&2.1e+4 & 1.2e+7&6.9e+6&4.8e+6 & 5.9e+5&3.4e+5&2.1e+5'
)

def triple_str(triple):
    red = 'F8CECC'    # highest
    yellow = 'FFF2CC' # medium
    green = 'D5E8D4'  # lowest
    first, second, third = triple
    ranking = [(first, 1), (second, 2), (third, 3)]
    ranking = sorted(ranking, key = lambda tupl : float(tupl[0]))
    colors = [green, yellow, red]

    ranking = [('\cellcolor[HTML]{' + color + '}{\color[HTML]{000000}' + val + '}', idx) for color, (val, idx) in zip(colors, ranking)]
    ranking = sorted(ranking, key = lambda tuple: tuple[-1])
    ranking = [x for x, _ in ranking]
    ranking = '&'.join(ranking)
    return ranking
    

if __name__ == '__main__':
    # Debugging
    from pprint import PrettyPrinter
    pp = PrettyPrinter()

    t = TSTR2
    nums = [[triple.strip().split('&') for triple in line.split(' & ')] for line in t.split('\n')]
    # pp.pprint(nums)
    nums = [[triple_str(triple) for triple in row] for row in nums]
    # pp.pprint(nums)
    nums = '\n'.join([' & '.join(row) for row in nums])
    print(nums)
    