# world-cup-simulator
This project was developed to simulate various World Cup formats using an XML configuration file. It allows us to run simulations with traditional formats (group phase and knockout), as well as double elimination and other formats.

To calculate the team strength, we initially consider the Elo Rating, which can be found at https://www.eloratings.net/. The Elo Rating method is adopted by FIFA to calculate their official World Ranking. Moreover, it is widely used as a benchmark in the literature.

This framework operates in a straightforward manner. To begin, we need to configure a new format in the "world-cup-formats.xml" file. Each format is identified by an ID and a name. Within each format, we specify the number of teams participating in the tournament, the number of teams in each group, the rounds of the tournament, and the matches within each round.
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

The <round> tag contains an attribute called "func" that calls several pre-implemented functions. For instance, the "GroupPhase" function automates the group phase of the tournament. The "compute" function is responsible for calculating the results in the group phase and reorganizing the group structure. Additionally, the "DoubleElimination" function simulates a double elimination tournament with a fixed number of 48 teams (for the initial version).

Now, let's take a closer look at how to configure each match. Bellow, we can see a simple sample of <match> tag:
```xml
<match id="1" typeHome="p" groupHome= "0" teamHome="0" typeAway="p" groupAway= "0" teamAway="1"/>
```
