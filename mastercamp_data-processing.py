import pandas as pd 
import requests 
import icalendar 
from datetime import datetime as dt
from dateutil import tz
import numpy as np 
from datetime import timedelta
import random
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
from deap import algorithms, base, creator, tools


link = "https://www.myefrei.fr/api/public/student/planning/AcSWV5Tr0fs9ulk5_8-XKQ"
density = [2,2,3,5,1,1,4,3,4,1,2,4]
difficulty = [2,2,3,5,0,0,4,2,4,0,3,5] 

class RandomPlan:
    def __init__(self, plan, fitness):
        self.plan = plan
        self.fitness = fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __len__(self):
        return len(self.plan)
    
    def __getitem__(self, index):
        return self.plan[index]

    def __setitem__(self, index, value):
        self.plan[index] = value
        


def main(link, density,difficulty): 
    df1 = get_link_data(link)
    print("get data OK")
    classes_list = create_class_list(df1,density,difficulty)
    print("create class list OK")
    df_free = creating_df_free(df1)
    print("df_free  OK")
    df_timeslots = creating_df_timeslots(df_free)
    print("df_timeslots OK")
    revision_plan = evolution(df_timeslots,classes_list)


    





def get_link_data(link):
    data = requests.get(link).text
    # parse the iCalendar data into a Calendar object
    cal = icalendar.Calendar.from_ical(data)

    # iterate over the calendar events and extract the relevant information
    events = []
    for event in cal.walk('VEVENT'):
        events.append({
            'dtstart': event.get('DTSTART').dt,
            'dtend': event.get('DTEND').dt,
            'summary': str(event.get('SUMMARY')),
            'description': str(event.get('DESCRIPTION')),
            'location': str(event.get('LOCATION'))
        })

    # create a data frame from the extracted information
    df = pd.DataFrame(events)

    df['exam'] = df['location'].apply(lambda x: 1 if x == 'Efrei Salles multiples' else 0)
    df['name'] = df['summary'].str.split('(', expand=True)[0]

    #CHANGE THIS PART TO FROM TODAY TO THE END 
    df1 = df[df['dtstart'] < "2022-01-17"]
    
    return df1

def calculate_total_duration(class_name,df1):
    filtered_df = df1[df1['name'] == class_name]
    total_duration = (filtered_df['dtend'] - filtered_df['dtstart']).sum().total_seconds() / 3600
    return total_duration

# Function to calculate hours per week based on the formula
def calculate_hours(row):
    # Define the weights
    density_weight = 0.4
    difficulty_weight = 0.4
    importance_weight = 0.2
    hours_per_week = round((density_weight * row['density'] +
                      difficulty_weight * row['difficulty'] +
                      importance_weight * row['importance']) / (density_weight + difficulty_weight + importance_weight))
    return hours_per_week


def create_class_list(df1,density,difficulty):
    liste_cours1 = df1.groupby(by='name').count()
    liste_cours1 = liste_cours1.drop(['dtstart','dtend','description','location','summary'],axis=1)
    liste_cours1.columns = ['occurence']
    liste_cours1 = liste_cours1[liste_cours1['occurence'] > 1]
    
    total = liste_cours1['occurence'].sum()
    liste_cours1['importance %'] = liste_cours1['occurence']*100/total
    liste_cours1 = liste_cours1.reset_index()
    
    filtered = df1[df1['exam'] == 1].groupby('name')['dtstart'].first().reset_index()
    filtered.columns = ['name','date exam']
    liste_finale = pd.merge(liste_cours1, filtered, on='name')
    #Group the data in df1 by 'name' and get the minimum date for each group
    first_class_dates = df1.groupby('name')['dtstart'].min()

    # Create a new column 'first class' in liste_finale and fill it with the corresponding first class dates
    liste_finale['first class'] = liste_finale['name'].map(first_class_dates)
    # Apply the function to each row of the 'liste_finale' DataFrame to calculate the total duration
    liste_finale['total_duration'] = liste_finale['name'].apply(lambda x: calculate_total_duration(class_name=x,df1=df1))
    # Calculate the number of weeks between 'first class' and 'date exam'
    liste_finale['weeks'] = np.ceil((liste_finale['date exam'] - liste_finale['first class']) / pd.Timedelta(days=7))
    liste_finale = liste_finale.sort_values('occurence', ascending= False,ignore_index=True)
    
    liste_finale['importance'] = pd.NA

    for i in range(len(liste_finale)):
        if i == 0:
            liste_finale.at[i, 'importance'] = 5  # Set importance as 5 for the first row
        else:
            diff = liste_finale.at[i-1, 'occurence'] - liste_finale.at[i, 'occurence']

            if diff < 2:
                liste_finale.at[i, 'importance'] = max(liste_finale.at[i-1, 'importance'] , 1)  # Set importance as previous row's importance if difference is less than 2
            else:
                liste_finale.at[i, 'importance'] = max(liste_finale.at[i-1, 'importance'] - 1,1)  # Set importance as previous row's importance - 1 if difference is 2 or more

    print(len(liste_finale))
    liste_finale['density'] = density
    liste_finale['difficulty'] = difficulty
    liste_finale['Hours per Week'] = liste_finale.apply(calculate_hours, axis=1)
    liste_finale['week exam'] = liste_finale['date exam'].dt.isocalendar().week
    liste_finale['first week'] = liste_finale['first class'].dt.isocalendar().week

    return liste_finale


def creating_df_free(df1):
    # Step 1: Convert "dtstart" and "dtend" columns to datetime objects
    df1['dtstart'] = pd.to_datetime(df1['dtstart'])
    df1['dtend'] = pd.to_datetime(df1['dtend'])

    # Step 2: Sort the dataframe by "dtstart" column
    df1 = df1.sort_values('dtstart')

    # Step 3: Find gaps between consecutive events
    free_slots = []
    previous_end = None

    for index, row in df1.iterrows():
        if previous_end is None:
            previous_end = row['dtend']
        else:
            if row['dtstart'] > previous_end:
                free_slots.append({
                    'dtstart': previous_end,
                    'dtend': row['dtstart'],
                    'summary': 'Free Time',
                    'description': '',
                    'location': '',
                    'exam': '',
                    'name': ''
                })
            previous_end = max(previous_end, row['dtend'])

    # Step 4: Create a new dataframe for free time slots
    df_free = pd.DataFrame(free_slots)
    return df_free 

def creating_df_timeslots(df_free):
    # Create an empty DataFrame for df_timeslots
    df_timeslots = pd.DataFrame(columns=df_free.columns)

    ## Define sleeping and wakeup times
    sleeping_time = pd.to_datetime('23:00', format='%H:%M').time()
    wakeup_time = pd.to_datetime('08:00', format='%H:%M').time()

    # Iterate over each row in df_free
    for index, row in df_free.iterrows():
        dtstart = row['dtstart']
        dtend = row['dtend']
        hours = (dtend - dtstart).total_seconds() / 3600  # Calculate the number of hours between dtstart and dtend
    
        if hours < 2:
            continue  # Skip creating rows if the time difference is less than 2 hour
    
        # Generate rows with incremented time intervals
        for i in range(int(hours)):
            current_start = dtstart + timedelta(hours=i)
            current_end = dtstart + timedelta(hours=i+1)
        
            # Check if the current time slot falls within the sleeping time range
            if current_start.time() >= sleeping_time or current_end.time() <= wakeup_time:
                continue  # Skip the row if it falls within the sleeping time range
        
            new_row = row.copy()
            new_row['dtstart'] = current_start
            new_row['dtend'] = current_end
            df_timeslots = df_timeslots.append(new_row, ignore_index=True)
            df_timeslots['weeks'] = df_timeslots['dtstart'].dt.isocalendar().week
            first_week_value = df_timeslots.iloc[0]['weeks']
    for i in range(len(df_timeslots)): 
        #print(df_timeslots.loc[i,'weeks'])
        if df_timeslots.loc[i,'weeks']>=first_week_value: 
            df_timeslots.loc[i,'weeks_relative'] = df_timeslots.loc[i,'weeks'] - first_week_value
        else:
            df_timeslots.loc[i,'weeks_relative'] = df_timeslots.loc[i,'weeks'] + (52-first_week_value)

    df_timeslots['weeks_relative'] = df_timeslots['weeks_relative'].astype(int)
    return df_timeslots

def add_week_day(random_plan): 
    random_plan['day']=0
    random_plan['day_of_week'] = random_plan['dtstart'].dt.day_name()

    for j in range(0, len(random_plan)): 
        if j == 0: 
            random_plan.loc[j, 'day'] = 0
        else: 
            if random_plan.loc[j, 'dtstart'].date() == random_plan.loc[j-1, 'dtstart'].date():
                random_plan.loc[j, 'day'] = random_plan.loc[j-1, 'day']
            else: 
                random_plan.loc[j, 'day'] = random_plan.loc[j-1, 'day'] + 1
    return random_plan

def random_generation(df_timeslots,classes_list):
    max_week = df_timeslots['weeks_relative'].max()
    df_revisions = pd.DataFrame()
    
    for i in range(0, max_week + 1): 
        df_test = df_timeslots[df_timeslots['weeks_relative'] == i]
    
        for j in range(0, len(classes_list)):
            compteur = int(classes_list.loc[j, 'Hours per Week'])
            first_class = int(classes_list.loc[j, 'first week'])
            exam_week = int(classes_list.loc[j, 'week exam'])
            total_weeks = int(classes_list.loc[j, 'weeks'])
            current_week = df_test['weeks'].max()
            
            if current_week >= first_class and current_week <= total_weeks + first_class:
                while compteur > 0: 
                    # Generate a random number within the range of the DataFrame's index
                    random_index = random.randint(0, len(df_timeslots) - 1)
                    
                    # Check if the randomly selected index has already been used
                    if random_index not in df_revisions.index:
                        # Select the row from df_timeslots using the randomly generated index
                        selected_row = df_timeslots.loc[random_index].copy()

                        if selected_row['dtstart'] > classes_list.loc[j, 'first class']:
                            # Modify the "summary" and "name" columns of the selected row
                            selected_row['summary'] = classes_list.loc[j, 'name']
                            selected_row['name'] = classes_list.loc[j, 'name']

                            # Add the selected row to df_revisions
                            df_revisions = df_revisions.append(selected_row)

                            # Decrement the counter
                            compteur -= 1

    return df_revisions

def creating_random_plan(df_timeslots,classes_list):
       # Create a list to store the randomly generated revision plans
    random_plans = []

   # Generate a random revision plan for each empty DataFrame
    for i in range(0,15):
        random_plan = random_generation(df_timeslots,classes_list)
        random_plan = random_plan.sort_values('dtstart')
        random_plan.reset_index(drop=True, inplace=True)
        total_days = random_plan['dtstart'].dt.date.nunique()
        random_plan = add_week_day(random_plan)
        plan = RandomPlan(random_plan,0)
        random_plans.append(plan)

    return random_plans


def calculate_hours_in_a_row(random_plan):
    
    revision_plan = random_plan.plan
    revision_plan = revision_plan.sort_values('dtstart')
    revision_plan = add_week_day(revision_plan)
    grouped = revision_plan.groupby('day')
    total_days = len(grouped)
    
    exceed_limit_days = 0

    for day, day_schedule in grouped:
        consecutive_hours_count = 0
        previous_end_time = None

        for index, row in day_schedule.iterrows():
            current_start_time = row['dtstart']
            
            if previous_end_time == current_start_time:
                consecutive_hours_count += 1
            else:
                consecutive_hours_count = 0

            if consecutive_hours_count > 4:
                exceed_limit_days += 1
                break

            previous_end_time = row['dtend']

    proportion = 1 - (exceed_limit_days / total_days)

    revision_plan = revision_plan[['dtstart','dtend','summary']]
    
    return proportion


def calculate_subjects_per_day(random_plan):

    revision_plan = random_plan.plan
    revision_plan = revision_plan.sort_values('dtstart')
    revision_plan = add_week_day(revision_plan)

    grouped = revision_plan.groupby('day')
    total_days = len(grouped)
    exceed_limit_days = 0

    for _, day_schedule in grouped:
        subjects_count = len(day_schedule['summary'].unique())

        if subjects_count > 2:
            exceed_limit_days += 1

    proportion = 1 - (exceed_limit_days / total_days)
    revision_plan = revision_plan[['dtstart','dtend','summary']]
    
    return proportion


def calculate_non_sundays(random_plan):
    
    revision_plan = random_plan.plan
    revision_plan = revision_plan.sort_values('dtstart')
    revision_plan = add_week_day(revision_plan)

    filtered_plan = revision_plan.drop_duplicates(subset='day')
    total_days = len(filtered_plan)
    non_sunday_days = len(filtered_plan[filtered_plan['day_of_week'] != 'Sunday'])

    proportion = non_sunday_days / total_days
    
    revision_plan = revision_plan[['dtstart','dtend','summary']]

    return proportion

"""def calculate_bulk_factor(individual):
    # Other factors and calculations in your fitness function
    revision_plan = individual.plan
    bulk_compteur = 0  # Initialize bulk factor
    non_bulk_compteur = 0

    # Calculate bulk factor
    for i in range(0,len(revision_plan) - 1):
        current_slot = revision_plan.iloc[i]
        next_slot = revision_plan.iloc[i + 1]
        
        # Check if the current and next slots are consecutive
        if current_slot['dtstart'].date() == next_slot['dtstart'].date():
            if current_slot['dtend'] == next_slot['dtstart']:
                bulk_compteur += 1
            else: 
                non_bulk_compteur += 1

    # Apply a weight to the bulk factor (adjust the weight according to importance)
    
    bulk_factor = bulk_compteur / (bulk_compteur+non_bulk_compteur)
   
    return bulk_factor"""

def calculate_freetime_usage_factor(individual, df_timeslots):
    if isinstance(individual.plan, str):
        revision_plan = pd.DataFrame(eval(individual.plan))
    else:
        revision_plan = individual.plan
    
    df_timeslots['dtstart'] = pd.to_datetime(df_timeslots['dtstart'])
    days = revision_plan['dtstart'].dt.date.unique()  # Get unique dates in the revision plan
    freetime_usage_count = 0  # Initialize count of days where freetime usage condition is met
    
    for day in days:
        revision_rows = revision_plan[revision_plan['dtstart'].dt.date == day]
        df_timeslots_rows = df_timeslots[df_timeslots['dtstart'].dt.date == day]
        
        if len(revision_rows) <= 0.7 * len(df_timeslots_rows):
            freetime_usage_count += 1
        
    freetime_usage_proportion = freetime_usage_count / len(days)
    
    return freetime_usage_proportion


def evaluate_plan(revision_plan, df_timeslots):
    proportion_hours_in_a_row = calculate_hours_in_a_row(revision_plan)
    proportion_subjects_per_day = calculate_subjects_per_day(revision_plan)
    proportion_non_sundays = calculate_non_sundays(revision_plan)
    #proportion_bulk_hours = calculate_bulk_factor(revision_plan)
    proportion_freetime_percentage = calculate_freetime_usage_factor(revision_plan, df_timeslots)
    total_fitness = 0.1 * proportion_hours_in_a_row + 0.2 * proportion_subjects_per_day + 0.3 * proportion_non_sundays + 0.4 * proportion_freetime_percentage

    revision_plan = revision_plan[['dtstart','dtend','summary']]
    # Return the fitness value as a tuple
    return (total_fitness,proportion_hours_in_a_row, proportion_subjects_per_day, proportion_non_sundays,proportion_freetime_percentage)

def mutate_plan(revision_plan, df_timeslots):
   
   # Calculate the number of rows to mutate
    num_mutations = int(len(revision_plan) * 0.2)
    
    # Generate unique random row indices
    mutation_indices = random.sample(range(len(revision_plan)), num_mutations)
    
    for random_row_index in mutation_indices:
        random_row = revision_plan.iloc[random_row_index]

        # Get the week of the randomly selected row
        selected_week = random_row['dtstart'].week

        # Filter df_timeslots to find rows in the same week
        available_slots = df_timeslots[df_timeslots['dtstart'].dt.week == selected_week]
    
        if len(available_slots) > 0:
            # Randomly select a row from the available slots
            selected_slot = random.choice(available_slots.index)

            # Check if dtstart and dtend already exist in the offspring plan
            existing_slots = revision_plan['dtstart'].values.tolist() + revision_plan['dtend'].values.tolist()
            while (available_slots.loc[selected_slot, 'dtstart'] in existing_slots) or (available_slots.loc[selected_slot, 'dtend'] in existing_slots):
                selected_slot = random.choice(available_slots.index)

            # Update the dtstart and dtend columns of the offspring plan
            revision_plan.at[random_row_index, 'dtstart'] = available_slots.loc[selected_slot, 'dtstart']
            revision_plan.at[random_row_index, 'dtend'] = available_slots.loc[selected_slot, 'dtend']
    
    
    
    return revision_plan

def crossover_plan(parent1, parent2):
    # Get the column names of the parents' dataframes
    parent1_columns = parent1.columns.tolist()
    parent2_columns = parent2.columns.tolist()

    # Convert the parents' plans to lists
    parent1_list = parent1.values.tolist()
    parent2_list = parent2.values.tolist()

    # Perform the crossover by swapping the elements beyond the crossover point
    crossover_point = random.randint(1, min(len(parent1_list), len(parent2_list)) - 1)
    offspring1_list = parent1_list[:crossover_point] + parent2_list[crossover_point:]
    offspring2_list = parent2_list[:crossover_point] + parent1_list[crossover_point:]

    # Create new offspring dataframes using the revised lists and column names
    offspring1 = pd.DataFrame(offspring1_list, columns=parent1_columns)
    offspring2 = pd.DataFrame(offspring2_list, columns=parent2_columns)

    return offspring1, offspring2

def evolution(df_timeslots,classes_list):
    random_plans = creating_random_plan(df_timeslots,classes_list)

    # Define the DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", pd.DataFrame, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,random_generation)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_plan)
    toolbox.register("mate", crossover_plan)
    toolbox.register("mutate", mutate_plan)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population_size = 10
    num_generations = 7


    # Create the initial population
    i=0
    print("i = " +str(i))
    population = random_plans
    for i in range(0,len(population)):
        population[i].plan = population[i].plan[['dtstart','dtend','summary']]

    """for i in range(0,len(population)):
        print(len(population[i].plan))"""

    best_fitness_list = []


    # Evaluate the fitness of the initial population
    fitness_values = list(map(lambda ind: toolbox.evaluate(ind, df_timeslots), population))
    for ind, fit in zip(population, fitness_values):
        ind.fitness = fit

    best_fitness = max(population, key=lambda x: x.fitness[0]).fitness[0]
    best_fitness_list.append(best_fitness)

    # Perform the evolution
    cxpb = 0.5  # Probability of crossover
    mutpb=0.2
    num_elites = 4


    for generation in range(num_generations):
        print("**************************************")
        print(f"Generation {generation+1}")
        print("**************************************")


        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
    
        # Convert offspring from tuple to list
        offspring = list(offspring)
        #print(offspring[0])
        #print(offspring[0].fitness)
        print("Len of offspring before the for = " + str(len(offspring)))

        """# Evaluate the fitness of the offspring
        for i in range(0, len(offspring)):
            offspring[i].fitness = toolbox.evaluate(offspring[i])
            print("Column names:")
            print(offspring[i].plan.columns.tolist())

            print("Length of the dataframe:")
            print(len(offspring[i].plan))
            print(offspring[i].plan.head(2))"""


        print("evaluation OK Crossover starting")

        # Apply crossover and mutation on the offspring
        
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1].plan, offspring[i].plan = toolbox.mate(offspring[i - 1].plan, offspring[i].plan)
                if hasattr(offspring[i - 1], 'fitness'):
                    del offspring[i - 1].fitness
                if hasattr(offspring[i], 'fitness'):
                    del offspring[i].fitness
        
        print("Crossover OK ")

    

        mutation_count = max(1, int(mutpb * len(offspring)))  # Number of individuals to mutate

        # Randomly select mutation_count individuals for mutation
        mutation_indices = random.sample(range(len(offspring)), mutation_count)
        #print(len(mutation_indices))
        # Perform mutation on the selected individuals
        for i in mutation_indices:
            offspring[i].plan = toolbox.mutate(offspring[i].plan,df_timeslots)
            if hasattr(offspring[i], 'fitness'):
                del offspring[i].fitness
        offspring[0].plan
        print("Mutation OK ")

        #CHECK PRINT
        """for i in range(0,len(offspring)):
            print("Column names:")
            print(offspring[i].plan.columns.tolist())

            print("Length of the dataframe:")
            print(len(offspring[i].plan))
            print(offspring[i].plan.head(2))"""

        # Evaluate the fitness of the offspring
        fitness_values = list(map(lambda ind: toolbox.evaluate(ind, df_timeslots), offspring))
        for ind, fit in zip(offspring, fitness_values):
            ind.fitness = fit
    
        print("Evaluation 2  OK ")

        # Selection for the next generation
        elites = []
        elites = tools.selBest(population, num_elites)

        # Sort the offspring based on fitness values (in descending order)
        offspring.sort(key=lambda x: x.fitness[0], reverse=True)

        # Identify the elite individuals from the current population
        elites = tools.selBest(population, num_elites)

        # Perform selection on the remaining individuals (non-elites) to create the next generation
        offspring = toolbox.select(offspring, len(population) - num_elites)

        # Replace the current population with the offspring (including elites)
        population[::] = elites + offspring

        #population[::] = offspring

        # Print the best fitness value in the current generation
        best_fitness = max(population, key=lambda x: x.fitness[0])
        best_fitness_list.append(best_fitness.fitness[0])
        print(f"Best Fitness: {best_fitness.fitness}")

    # Print the best individual (random plan) in the final population
    best_individual = max(population, key=lambda x: x.fitness[0])
    print("Best Individual:")

    print(best_individual.fitness)
    print(best_individual.plan)

    
    # Plot the fitness values
    plt.plot(range(1, num_generations + 2), best_fitness_list)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progression")
    plt.show()
    return best_individual.plan


main(link,density,difficulty)