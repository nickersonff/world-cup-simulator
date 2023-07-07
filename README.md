# world-cup-simulator
This project was developed to simulate various World Cup formats using an XML configuration file. It allows us to run simulations with traditional formats (group phase and knockout), as well as double elimination and other formats.

To calculate the team strength, we initially consider the Elo Rating, which can be found at https://www.eloratings.net/. The Elo Rating method is adopted by FIFA to calculate their official World Ranking. Moreover, it is widely used as a benchmark in the literature.

This framework operates in a straightforward manner. To begin, we need to configure a new format in the "world-cup-formats.xml" file (present in content path). Each format is identified by an ID and a name. Within each format, we specify the number of teams participating in the tournament, the number of teams in each group, the rounds of the tournament, and the matches within each round.
Here is simple example of the XML configuration file:

```xml
<?xml version="1.0"?>
<formats>
	<format id="0" name="Cup48_4groups">
		<teams>48</teams>
		<groups>4</groups>
		<rounds>	
			<round id="0" func="GroupPhase"/>
			<round id="1"> 
				<match id="72" typeHome="p" groupHome= "1" teamHome="0" typeAway="p" groupAway= "13" teamAway="0"/>
				<match id="73" typeHome="p" groupHome= "2" teamHome="1" typeAway="p" groupAway= "6" teamAway="1"/>
				<match id="74" typeHome="p" groupHome= "6" teamHome="0" typeAway="p" groupAway= "5" teamAway="1"/>				
			</round>
			<round id="2">
				<match id="88" typeHome="w" matchHome= "72" typeAway="w" matchAway="73"/>
				<match id="89" typeHome="w" matchHome= "74" typeAway="w" matchAway="75"/>
				<match id="90" typeHome="w" matchHome= "76" typeAway="w" matchAway="77"/>
			</round>
		</rounds>
	</format>
</formats>
```

The `<round>` tag contains an attribute called "func" that calls several pre-implemented functions. For instance, the "GroupPhase" function automates the group phase of the tournament. The "compute" function is responsible for calculating the results in the group phase and reorganizing the group structure. Additionally, the "DoubleElimination" function simulates a double elimination tournament with a fixed number of 48 teams (for the initial version).

Now, let's take a closer look at how to configure each match. Bellow, we can see a simple sample of `<match>` tag:
```xml
<match id="1" typeHome="p" groupHome= "0" teamHome="0" typeAway="p" groupAway= "0" teamAway="1"/>
```
Note that the `<match>` tag contains attributes for the home team and away team of the match, as well as their indexes in the group structure. Additionally, there are typeHome and typeAway attributes that specify how the framework searches for each team. The possible values for these attributes are:

`p` -> search for the team's position in the group structure.
`w` -> select the winner of another match.
`l` -> select the loser of another match.

In the previous example, for the match with an ID of 1, the framework will search for the team at index 0 in group index 0, and the team at index 1 in group index 0 to determine the home and away teams for the match. In the next example, the framework will construct a match between the winner of the match with an ID of 72 and the winner of the match with an ID of 73.
```xml
<match id="2" typeHome="w" matchHome= "72" typeAway="w" matchAway="73"/>
```

Another important aspect to understand is how the group structure works. Initially, the group structure consists of a "single group" with an index of 0. However, the framework processes each group separately based on the number of teams per group. For instance, in a tournament with 48 teams and 4 teams per group, the framework will consider teams with indices 0 to 3 as one group, indices 4 to 7 as another group, and so on. Therefore, in the previous match example, it would involve the team with an index of 0 against the team with an index of 1, both belonging to the same group (the first group). This first group is typically utilized for the group phase of the tournament.

Then, when we use the "compute" function, all results are calculated and the framework will create one ordered group for each group of the tournament and an aditional group for the best 3rd-placed teams. To illustrate, let's consider a tournament with 48 teams and 4 teams per group, resulting in a total of 12 groups. Initially, our group structure will consist of "only one" group during the group phase. However, after all matches are played and the compute function is executed (refer to the "Cup48_3groups" format in the world-cup-formats.xml file), thirteen new groups are created within the group structure. We can then construct matches using these "new groups", as exemplified below:
```xml
<match id="48" typeHome="p" groupHome= "1" teamHome="0" typeAway="p" groupAway= "2" teamAway="1"/>
<match id="48" typeHome="p" groupHome= "2" teamHome="0" typeAway="p" groupAway= "13" teamAway="2"/>
```

Now, we can proceed with creating our simulations to measure various metrics, as demonstrated in the Test files available in the source code.
