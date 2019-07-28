import numpy as np
import random
import time
import pickle


class Env:
    state = None
    done = False

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

        return self.state

    def step(self, player_id, position):
        if (self.state[position] != 0):
            raise Exception("Invalid move !")

        self.state[position] = player_id

        if self.win():
            reward = 1
            self.done = True
        elif self.draw():
            reward = 0
            self.done = True
        else:
            reward = 0
            self.done = False

        return self.state, reward, self.done

    def win(self):
        # check each row
        for row in range(3):
            if ((self.state[row][0] != 0) & (self.state[row][0] == self.state[row][1] == self.state[row][2])):
                return True

        # check each column
        for column in range(3):
            if ((self.state[0][column] != 0) & (self.state[0][column] == self.state[1][column] == self.state[2][column])):
                return True

        # check left to right diagonal
        if ((self.state[0][0] != 0) & (self.state[0][0] == self.state[1][1] == self.state[2][2])):
            return True

        # check right to left diagonal
        if ((self.state[0][2] != 0) & (self.state[0][2] == self.state[1][1] == self.state[2][0])):
            return True

        return False

    def draw(self):
        for row in self.state:
            for cell in row:
                if cell == 0:
                    return False

        return True

    def get_available_actions(self, state):
        available_actions = []

        # loop through each cell
        for row in range(3):
            for column in range(3):
                # empty cell
                if state[row][column] == 0:
                    available_actions.append((row, column))

        return available_actions


class Agent:
    id = None
    state_table = None
    name = None
    epsilon = 0.35
    previous_state_action = None
    learning_rate = 0.1
    error_mean = 0.5

    def __init__(self, id, table):
        self.id = id
        self.state_table = table
        self.name = "agent_" + str(id)

    def choose_action(self, state):
        percieved_state = np.zeros(state.shape)
        available_actions = []

        # loop through each cell
        for row in range(3):
            for column in range(3):
                # empty cell
                if state[row][column] == 0:
                    available_actions.append((row, column))

                # own cell, set to 1
                elif state[row][column] == self.id:
                    percieved_state[row][column] = 1

                # opponent cell, set to 2
                else:
                    percieved_state[row][column] = 2

        key = percieved_state.tostring()

        # if state exists in table, pull a list action values for this state from the table
        if key in self.state_table:
            action_values = self.state_table[key]

        # else, add the state to the table and create a list action values for this state
        else:
            action_values = {}
            for action in available_actions:
                action_values[action] = 0.5

            # insert this state into table
            self.state_table[key] = action_values

        # e-greedy
        if random.random() >= self.epsilon:
            # get the action tuple that has the maximum value
            action = max(action_values, key=(lambda action: action_values[action]))

            # greedy move: update previous state action value using current state action value
            self.update_previous_state_action_value(action_values[action], False)

        else:
            action = random.choice(list(action_values.items()))[0]

        # store the last state, action
        self.previous_state_action = (percieved_state, action)

        return action

    def update_previous_state_action_value(self, target_value, reset_previous_state_action):
        if self.previous_state_action != None:
            previous_state = self.previous_state_action[0]
            previous_action = self.previous_state_action[1]

            key = previous_state.tostring()
            action_values = self.state_table[key]
            value = action_values[previous_action]
            action_values[previous_action] = action_values[previous_action] + self.learning_rate * (target_value - value)
            self.state_table[key] = action_values

            #print(self.name, "update action", previous_action, "from", value, "to", action_values[previous_action])
            if reset_previous_state_action:
                self.previous_state_action = None


if __name__ == "__main__":
    file = 'state_table.p'
    try:
        with open(file, 'rb') as fp:
            state_table = pickle.load(fp)
            print("state_table loaded from", file)
    except:
        print("state_table.json not found, initializing state_table")
        state_table = {}

    env = Env()
    agent_1 = Agent(1, state_table)
    agent_2 = Agent(2, state_table)
    previous_player = None
    current_player= None
    episodes = 1000
    learn = True

    if learn:
        # agent learning
        print("Learning begin")
        for e in range(episodes):
            print("GAME ", e)
            current_player = agent_1
            done = False
            state = env.reset()

            while not done:
                action = current_player.choose_action(state)
                next_state, reward, done = env.step(current_player.id, action)

                if done:
                    # game is won by current player, means previous player made the wrong move
                    if reward == 1:
                        #print("Player", current_player.id, " wins !!\n")
                        current_player.update_previous_state_action_value(1, True)
                        previous_player.update_previous_state_action_value(-1, True)

                        break
                    # game draw
                    else:
                        #print("Game is draw! \n")
                        current_player.update_previous_state_action_value(-0.5, True)
                        previous_player.update_previous_state_action_value(-0.5, True)

                        break

                # game is not over yet, switch to next player
                state = next_state

                if current_player == agent_1:
                    current_player = agent_2
                    previous_player = agent_1
                else:
                    current_player = agent_1
                    previous_player = agent_2

        print("Learning complete")

        # save state table object into p file (python native object file)
        with open(file, 'wb') as fp:
            pickle.dump(state_table, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print("state_table saved ", )

    # test play with human
    human_id = 3
    ai = agent_1
    ai.epsilon = 0.01  # suppress exploration
    current_player_id = human_id

    while(True):
        print("New Game:")
        state = env.reset()
        done = False
        print(state)

        while not done:
            current_player_name = "Your" if current_player_id == human_id else "Computer's"
            print(current_player_name, "turn:")
            available_moves = env.get_available_actions(state)
            print("available moves:", available_moves)

            if current_player_id == human_id:
                valid_input = False
                while not valid_input:
                    action = input("Please make a move, example enter 1 1 to choose (1, 1): ")
                    action = tuple(int(x) for x in action.split())
                    if action in available_moves:
                        valid_input = True
            else:
                time.sleep(0.3)
                action = ai.choose_action(state)
                print("Computer choose", action)

            state, reward, done = env.step(current_player_id, action)
            print(state)

            if done:
                if reward == 1:
                    if current_player_id == ai.id:
                        print("Computer win !")
                        ai.update_previous_state_action_value(1, True)
                    else:
                        print("You win !")
                        ai.update_previous_state_action_value(-1, True)
                elif reward == 0:
                    print("Game is draw !")
                    if current_player_id == ai.id:
                        ai.update_previous_state_action_value(0, True)

            current_player_id = human_id if current_player_id != human_id else ai.id
            print()