import java.awt.Color;
import java.util.List;

import burlap.behavior.singleagent.*;
import burlap.domain.singleagent.gridworld.*;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.*;
import burlap.oomdp.singleagent.common.*;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.singleagent.learning.*;
import burlap.behavior.singleagent.learning.tdmethods.*;
import burlap.behavior.singleagent.planning.*;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.*;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.oomdp.visualizer.Visualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.*;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.oomdp.singleagent.common.VisualActionObserver;

public class BasicBehavior {

    GridWorldDomain             gwdg;
    Domain                      domain;
    StateParser                 sp;
    RewardFunction              rf;
    TerminalFunction            tf;
    StateConditionTest          goalCondition;
    State                       initialState;
    DiscreteStateHashFactory    hashingFactory;

    public static void main(String[] args) {
        BasicBehavior example = new BasicBehavior();
        String outputPath = "output/";

        //uncomment the example you want to see (and comment-out the rest)

        //example.ValueIterationExample(outputPath);
        //example.PolicyIterationExample( outputPath );
        //example.QLearningExample(outputPath);
        example.SarsaLearningExample(outputPath);
        //example.experimenterAndPlotter();

        //run the visualizer (only use if you don't use the experiment plotter example)
        example.visualize(outputPath);
    }


    public BasicBehavior(){
        //create the domain
        //gwdg = new GridWorldDomain(25, 25);
        gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();
        //gwdg.makeEmptyMap();
        domain = gwdg.generateDomain();

        //create the state parser
        sp = new GridWorldStateParser(domain);

        //define the task
        rf = new UniformCostRF();
        tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));
        goalCondition = new TFGoalCondition(tf);

        //set up the initial state of the task
        initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(initialState, 0, 0);
        GridWorldDomain.setLocation(initialState, 0, 10, 10);

        //set up the state hashing system
        hashingFactory = new DiscreteStateHashFactory();
        hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT,
                domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);


        //add visual observer
        VisualActionObserver observer = new VisualActionObserver(domain,
            GridWorldVisualizer.getVisualizer(gwdg.getMap()));
        ((SADomain)this.domain).setActionObserverForAllAction(observer);
        observer.initGUI();
    }


    public void visualize(String outputPath){
        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
        EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v,
                                domain, sp, outputPath);
    }

    public void ValueIterationExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        long start = System.currentTimeMillis();

        OOMDPPlanner planner = new ValueIteration(domain, rf, tf, 0.99, hashingFactory,
                                0.001, 10000);
        planner.planFromState(initialState);

        long elapsedTimeMillis  = System.currentTimeMillis() - start;
        System.out.println( "Value Iteration took " + elapsedTimeMillis/1000F + " seconds\n");

        //create a Q-greedy policy from the planner
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult", sp);

        //visualize the value function and policy
        this.valueFunctionVisualize((QComputablePlanner)planner, p);
    }

    public void PolicyIterationExample(String outputPath ){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        long start = System.currentTimeMillis();

        OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, 0.99, hashingFactory,
                                0.001, 100, 100);
        planner.planFromState(initialState);

        long elapsedTimeMillis  = System.currentTimeMillis() - start;
        System.out.println( "Policy Iteration took " + elapsedTimeMillis/1000F + " seconds\n");

        //create a Q-greedy policy form the planner
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult", sp);

        //visualize the value function and policy
        this.valueFunctionVisualize((QComputablePlanner)planner, p);
    }

    public void QLearningExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //discount= 0.99; initialQ=0.0; learning rate=0.9
        LearningAgent agent = new QLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);

        //run learning for 100 episodes
        for(int i = 0; i < 100; i++){
            EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
            ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
            System.out.println(i + ": " + ea.numTimeSteps());
        }
    }

    public void SarsaLearningExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //discount= 0.99; initialQ=0.0; learning rate=0.5; lambda=1.0
        LearningAgent agent = new SarsaLam(domain, rf, tf, 0.99, hashingFactory,
                        0., 0.5, 1.0);

        //run learning for 100 episodes
        for(int i = 0; i < 100; i++){
            EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
            ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
            System.out.println(i + ": " + ea.numTimeSteps());
        }
    }

    public void valueFunctionVisualize(QComputablePlanner planner, Policy p){
        List <State> allStates = StateReachability.getReachableStates(initialState,
            (SADomain)domain, hashingFactory);
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
            GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);

        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
            GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);

        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
        gui.setSpp(spp);
        gui.setPolicy(p);
        gui.setBgColor(Color.GRAY);
        gui.initGUI();
    }

    public void experimenterAndPlotter(){
        //custom reward function for more interesting results
        final RewardFunction rf = new GoalBasedRF(this.goalCondition, 5., -0.1);

        LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

            @Override
            public String getAgentName() {
                return "SARSA";
            }

            @Override
            public LearningAgent generateAgent() {
                return new SarsaLam(domain, rf, tf, 0.99, hashingFactory, 0.0, 0.1, 1.);
            }
        };

        StateGenerator sg = new ConstantStateGenerator(this.initialState);

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter((SADomain)this.domain,
            rf, sg, 2, 100, sarsaLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
            TrialMode.MOSTRECENTANDAVERAGE,
            PerformanceMetric.CUMULTAIVEREWARDPEREPISODE,
            PerformanceMetric.AVERAGEEPISODEREWARD);

        long start = System.currentTimeMillis();

        exp.startExperiment();

        long elapsedTimeMillis  = System.currentTimeMillis() - start;
        System.out.println( "Experiment took " + elapsedTimeMillis/(60*1000F) + " minutes\n");

        exp.writeStepAndEpisodeDataToCSV("expData");

    }
}
