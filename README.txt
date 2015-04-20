Instructions for running code: 

Make sure that you are in current directory where this file is placed. 

In console, run 
    javac -cp lib/* BasicBehavior.java && java -cp lib/*:. BasicBehavior

This will compile and run code to replicate experiments. 

For specific experiments, such as running QLearning open BasicBehavior.java, 
in main method instructions for running each experiment is found in comments. 

To change the domains form large to small state size, toggle between 
GridWorldDomain constructors and use the setMapToFourRooms function for 11x11
and makeEmptyMap for 25x25
