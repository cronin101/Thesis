import numpy as np


ASYMPTOTIC_FITNESS = 200
BREEDING_SEASON_LENGTH = 60
CRITICAL_RESERVE_AMT, MAX_RESERVE_AMT = 0, 100
RESERVE_STATES = (MAX_RESERVE_AMT - CRITICAL_RESERVE_AMT) + 1
STARTING_RESERVES = MAX_RESERVE_AMT * 0.25


class Patch(object):
    def __init__(self, index, food_value=0, foraging_success_rate=0, mortality_rate=0, visiting_cost=3):
        self.index, self.food_value, self.foraging_success_rate = index, food_value, foraging_success_rate
        self.mortality_rate, self.visiting_cost = mortality_rate, visiting_cost

    def post_visit_fitness(self, reserves, t):
        fail_r = max(reserves - self.visiting_cost, CRITICAL_RESERVE_AMT)
        success_r = min(reserves + self.food_value - self.visiting_cost, MAX_RESERVE_AMT)
        return (1 - self.mortality_rate) * ((fitness[success_r, (-1)+t+1] * self.foraging_success_rate) +
            (fitness[fail_r, (-1)+t+1] * (1 - self.foraging_success_rate)))


class RefugePatch(Patch):
    def __init__(self, index):
        return Patch.__init__(self, index, mortality_rate=0, food_value=0, foraging_success_rate=0)


PATCHES = [ Patch(index=1, food_value=8, foraging_success_rate=0.5, mortality_rate = 0.05),
            Patch(index=2, food_value=20, foraging_success_rate=0.2, mortality_rate = 0.05),
            RefugePatch(index=3) ]

def fitness_function(reserve_level):
    surplus = reserve_level - CRITICAL_RESERVE_AMT
    return ASYMPTOTIC_FITNESS * (surplus / (surplus + STARTING_RESERVES))

fitness = np.zeros((RESERVE_STATES, BREEDING_SEASON_LENGTH))
decision = np.zeros((RESERVE_STATES, BREEDING_SEASON_LENGTH), dtype=np.int)

for r in range(CRITICAL_RESERVE_AMT, MAX_RESERVE_AMT+1): # Calculate end conditions
  fitness[r, (-1)+BREEDING_SEASON_LENGTH] = fitness_function(r)

for t in range(BREEDING_SEASON_LENGTH - 1, 0, -1): # Work backwards to find optimal choices
    for reserves in range(CRITICAL_RESERVE_AMT + 1, MAX_RESERVE_AMT+1):
        optimal_choice = max(PATCHES,key=lambda(patch): (patch.post_visit_fitness(reserves, t), -patch.index))
        decision[reserves, (-1)+t] = optimal_choice.index
        fitness[reserves, (-1)+t] = optimal_choice.post_visit_fitness(reserves, t)

np.savetxt("decisions.csv", np.asarray(decision), fmt='%d', delimiter=",")
