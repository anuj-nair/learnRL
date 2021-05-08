# Reinforcement Learning Foundations


## Introduction
Reinforcement Learning is a form of machine learning or one of the ways machines learn that involves interaction with the environment and learning from their own actions as opposed to history or previous data sets.

### Basics

* Agent
	Reinforcement learning agents observe and explore the environment learn

* State
	Position where an is Agent is, at a given period

* Environment
	Environment where a Agent observes to learn 

* Action
	Steps an Agent decides after observing a Environment

* Reward
	Score a Agent gets for taking Action

	|Correct Action|Wrong Action|
	|---|---|
	|Positive Reward|Negative Reward|

* Policy
	Strategy for deciding the best Action on particular State

* Goal
	Goal of the Agent from observing and learning from an Environment
	
	|Episodic Task|Continuing Task|
	|---|---|
	|These are tasks that have a defined goal or end point|These are tasks that don't have an end point and they continue forever|
	|Mostly solved by model-based methods|Mostly solved by model-free methods|

* Episode
	They are set of Actions 

* Discount Factor
	This is the Reward function for Continuing Task 

* Environment-Model

	|Unknown Environment|Known Environment|
	|---|---|
	|Model-Free|Model-Based|

* Exploration and Exploitation
	* Exploration
		Searches through the Environment and tries to understand it
	* Exploitation
		Saving information learned about the Environment
	* Best Strategy 
		* Initially
			Favor Exploration 
		* Lean towards 
			Exploration 
	
### Reinforcement Learning Problem
* Markov Decision Process (MDP)
	* Problem should Divided into
		* A set of States
		* A set of Actions
		* A set of Rewards
		* Goals
* Bellman Equation
	
	V<sub>π</sub>(s) = E<sub>π</sub> [ R<sub>t+1</sub> + γV<sub>π</sub> ( S<sub>t+1</sub> ) | S<sub>t</sub> = s ]
		
	* V<sub>π</sub>(s) 
		
		'V' is the State Value of a State 's' following a Policy 'π'
	
	* E<sub>π</sub>
	
		'E' is the expected State Value following Policy 'π'
	
	* R<sub>t+1</sub>
		
		'R' is the Reward of moving to a State '(t+1)'
	
	* γ
		
		'γ' is the Discount Factor which helps us favour recent Rewards, as opposed to older Rewards
	
	* V<sub>π</sub> ( S<sub>t+1</sub> ) 
		
		It is the State Value of the next State
	
	* S<sub>t</sub> = s  
		
		It implies that the whole equation is applicable to current state 's'

	|State Value Function|Action Value Function|
	|---|---|
	|Expected value of reward in a particular state| Expected value of the reward in a particular state given that the agent has taken an initial action|
	
### Q-Table

||State 1|State 2|State 3|State 4|State 5|State 6|
|---|---|---|---|---|---|---|
| **Forward** |||||||
| **Backward** |||||||
| **Right** |||||||
| **Left** ||||||Goal|

## Algorithms


|Monte Carlo|Temporal Difference|
|---|---|
|Updates Q-Table at the end of every episode|Updates Q-Table at every time step|
|High variance|Low variance|			
|Low bias|High bias|



### Monte Carlo Method
The Monte Carlo prediction uses Bellman equation to estimate the state and action value functions

#### Visits

|First-Visit Monte Carlo Prediction|Every-Visit Monte Carlo Prediction|
|---|---|
|Take only the first visit to a state into consideration|Take a state as a new state every time it is visited|

#### Bellman Equation

Q(S<sub>t</sub>,A<sub>t</sub>) = Q(S<sub>t</sub>,A<sub>t</sub>) +	α(G<sub>t</sub> - Q(S<sub>t</sub>,A<sub>t</sub>)) 


* Q(S<sub>t</sub>,A<sub>t</sub>)
	It is the Action Value function after an episode
* α
	Learning Rate
* G<sub>t</sub>
	It is the Total Reward gotten at the end of an Episode
- Q(S<sub>t</sub>,A<sub>t</sub>)) 




#### Monte Carlo Control Cycle
* Construct Q-table using equiprobable random policies
* Improve policy using Bellman equation
* Update Q-table

#### Additional
* Greedy Policies
	Policies that only select the best action for a given state all the time
* Epsilon-Greedy Policies
	Policies that select other action with respect to exploration and exploitation
* Incremental Mean
	Used to update the policy after every episode
	* Monte Carlo method without Incremental Mean:
		* Updates Q-table after multiple episodes
	* Monte Carlo method with Incremental Mean:
		* Updates Q-Table after an Episode
	
	|Episode|Reward|Action Values|
	|---|---|---|
	|1|2|2|
	|2|4|3|
	|3|6|4|
* Constant Alpha
	* Used to update the policy after every episode
	* Also known as Learning Rate
	* It's between 0-1




### Temporal Difference Methods
Temporal Difference methods exploit the Markov property by taking only the previous state into consideration when determining the value of the current state.
It is used for Continuing tasks

#### SARSA
* State -> Action -> Reward -> State -> Action -> Update Policy -> Select Next State
* It uses a Epsilon-Greedy Policy

##### Bellman Equation

Q(S<sub>t</sub>,A<sub>t</sub>) = Q(S<sub>t</sub>,A<sub>t</sub>) +	α(R<sub>t</sub> + γQ(S<sub>t+1</sub>,A<sub>t+1</sub>) - Q(S<sub>t</sub>,A<sub>t</sub>)) 

* Q(S<sub>t</sub>,A<sub>t</sub>)
	It is the Action Value function
* α
	Learning Rate
* (R<sub>t</sub> + γQ(S<sub>t+1</sub>,A<sub>t+1</sub>)
	It is the sum of the Reward of the Next State and the Action Value of the Next State

#### SARSAMAX (Q-Learning)
* State -> Action -> Reward -> State -> Update Policy -> Select Next Action
* For SARSAMAX, the action chosen for a state maximizes the action value for a state.
* It uses a Greedy Policy


##### Bellman Equation

Q(S<sub>t</sub>,A<sub>t</sub>) = Q(S<sub>t</sub>,A<sub>t</sub>) +	α(R<sub>t</sub> + γmaxQ(S<sub>t+1</sub>,A<sub>t+1</sub>) - Q(S<sub>t</sub>,A<sub>t</sub>)) 


#### Expected SARSA
* Uses the expected value of the State Action pair, where this Expected Value takes in account the probability that the agent selects each possible action from the next state. That is, there is an equal probability of selecting every action in the next state.
* Uses Epsilon-Greedy policy

## Modified Forms of Reinforcement

### Multiple Agent Reinforcement Learning

|Cooperative|Competitive|General|
|---|---|---|
|Working towards Common Goal|Working towards Different Goal|Learning in an open space without parters|

