# Table Detector

This is a simple tool used to automate table detection and extraction from plain text documents,
where tables are simple printed as loosely spaced / separated columns, with no formal delimiters,
like in a csv file for instance. An example is the text file below, where rows of the table are
nested among other non-table lines, and we are interested in extracting the first and second tables.


```txt
The CoronaVirus checklist

Doomsday supplies

Header0            Header1                  Header2  Header3($)
0000001            Canned food                  500        9,99
0000002            Toilet paper            20 packs       10,88
0000003            Gas mask                       8       90,00
0000004            Shotgun                        2      100,00
0000005            Mayonnaise                   5kg        0,99

Here is another table, showing how many eggs I am able to swallow
Eggs              Ability to swallow                         Am I dead?
0000              Nothing to swallow                                 No
0001              Difficult, but possible                            No
0002              Dangerous, choking hazard                          No
0003              I am dying, please stop                            No
0004              Reborn as the easter bunny                        Yes
```

The tool makes the following assumptions:
- Characters of the entries of each row in a table are similarly distributed, i.e. the columns of each
row are under each other
- Rows of each table are located at adjacent lines (which is the case for almost all tables)
- Tables are space-delimited (although we can change this quite easily to accomodate other formats, but this
seems to be the widest use-case)

A similarily matrix based on the distribution of spaces and characters are used to identify multiple
similar, adjacent lines, and these are then extracted. The table is then constructed by splitting
these lines along spaces.

Similarity matrix:

![alt text](https://raw.githubusercontent.com/whong92/FuzzyTableDetect/master/example.png "Similarity matrix")

Extracted tables:

![alt text](https://raw.githubusercontent.com/whong92/FuzzyTableDetect/master/example_out.png "Similarity matrix")
