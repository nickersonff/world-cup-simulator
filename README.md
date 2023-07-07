# world-cup-simulator
This project was developed to simulate various World Cup formats using an XML configuration file. It allows us to run simulations with traditional formats (group phase and knockout), as well as double elimination and other formats.

To calculate the team strength, we initially consider the Elo Rating, which can be found at https://www.eloratings.net/. The Elo Rating method is adopted by FIFA to calculate their official World Ranking. Moreover, it is widely used as a benchmark in the literature.

This framework operates in a straightforward manner. To begin, we need to configure a new format in the "world-cup-formats.xml" file. Each format is identified by an ID and a name. Within each format, we specify the number of teams participating in the tournament, the number of teams in each group, the rounds of the tournament, and the matches within each round.

The <round> tag contains an attribute called "func" that calls several pre-implemented functions. For instance, the "GroupPhase" function automates the group phase of the tournament. The "compute" function is responsible for calculating the results in the group phase and reorganizing the group structure. Additionally, the "DoubleElimination" function simulates a double elimination tournament with a fixed number of 48 teams (for the initial version).

Now, let's take a closer look at how to configure each match. 
