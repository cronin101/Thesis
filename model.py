import numpy as np


class Patch(object):
  def __init__(self, index, food_value, foraging_success_rate,
      mortality_rate=0, visiting_cost=3, is_refuge=False):
    self.index                 = index
    self.food_value            = food_value
    self.foraging_success_rate = foraging_success_rate
    self.mortality_rate        = mortality_rate
    self.visiting_cost         = visiting_cost
    self.is_refuge             = is_refuge


class RefugePatch(Patch):
  def __init__(self, index):
    return Patch.__init__(self, index, mortality_rate=0, food_value=0, foraging_success_rate=0, is_refuge=True)


# Inputs
BREEDING_SEASON_LENGTH = 60
CRITICAL_RESERVE_AMT, MAX_RESERVE_AMT = 0, 100
RESERVE_STATES = (MAX_RESERVE_AMT - CRITICAL_RESERVE_AMT) + 1
STARTING_RESERVES = MAX_RESERVE_AMT / 0.25

PATCHES = [
    Patch(index=1, food_value=10, foraging_success_rate=0.5, mortality_rate = 0.01),
    Patch(index=2, food_value=20, foraging_success_rate=0.25, mortality_rate = 0.01),
    RefugePatch(index=3)
  ]

ASYMPTOTIC_FITNESS = 200
def fitness_function(reserve_level):
  surplus = reserve_level - CRITICAL_RESERVE_AMT
  return ASYMPTOTIC_FITNESS * (surplus / (surplus + STARTING_RESERVES))

# Tables for DP
fitness = np.zeros((RESERVE_STATES, BREEDING_SEASON_LENGTH))
decision = np.zeros((RESERVE_STATES, BREEDING_SEASON_LENGTH), dtype=np.int)

# Calculate end conditions
for reserve_level in range(CRITICAL_RESERVE_AMT, MAX_RESERVE_AMT+1):
  fitness[reserve_level, BREEDING_SEASON_LENGTH-1] = fitness_function(reserve_level)

def patch_visiting_fitness(patch, reserve_level, timestep):
  if patch.is_refuge:
    reserves = max(reserve_level - patch.visiting_cost, CRITICAL_RESERVE_AMT)
    return (1 - patch.mortality_rate) * fitness[reserve_level, timestep]

  else:
    failure_reserves = reserve_level - patch.visiting_cost
    success_reserves = failure_reserves + patch.food_value
    failure_reserves = max(failure_reserves, CRITICAL_RESERVE_AMT)
    success_reserves = min(success_reserves, MAX_RESERVE_AMT)
    return (1 - patch.mortality_rate) * (
        fitness[success_reserves, timestep] * patch.foraging_success_rate +
        fitness[failure_reserves, timestep] * (1 - patch.foraging_success_rate))

# Work backwards to find optimal choices
for timestep in range(BREEDING_SEASON_LENGTH - 1, 0, -1):
  for reserve_level in range(CRITICAL_RESERVE_AMT, MAX_RESERVE_AMT+1):
    optimal_choice = max(PATCHES,
        key=lambda(patch): (patch_visiting_fitness(patch, reserve_level, timestep), -patch.index))

    decision[reserve_level, timestep-1] = optimal_choice.index

    fitness[reserve_level, timestep-1] = patch_visiting_fitness(
        optimal_choice, reserve_level, timestep)

np.savetxt("decisions.csv", np.asarray(decision), fmt='%d', delimiter=",")
