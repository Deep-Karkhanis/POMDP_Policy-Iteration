#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>
#include <chrono>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/PolicyIteration.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>

// RENDERING

// Special character to go back up when drawing.
std::string up =   "\033[XA";
// Special character to go back to the beginning of the line.
std::string back = "\33[2K\r";

void goup(unsigned x) {
    while (x > 9) {
        up[2] = '0' + 9;
        std::cout << up;
        x -= 9;
    }
    up[2] = '0' + x;
    std::cout << up;
}

void godown(unsigned x) {
    while (x) {
        std::cout << '\n';
        --x;
    }
}

const std::vector<std::string> prize {
    { R"(  ________  )" },
    { R"(  |       |\)" },
    { R"(  |_______|/)" },
    { R"( / $$$$  /| )" },
    { R"(+-------+ | )" },
    { R"(|       |/  )" },
    { R"(+-------+   )" }};

const std::vector<std::string> tiger {
    { R"(            )" },
    { R"(   (`/' ` | )" },
    { R"(  /'`\ \   |)" },
    { R"( /<7' ;  \ \)" },
    { R"(/  _､-, `,-\)" },
    { R"(`-`  ､/ ;   )" },
    { R"(     `-'    )" }};

const std::vector<std::string> closedDoor {
    { R"(   ______   )" },
    { R"(  /  ||  \  )" },
    { R"( |   ||   | )" },
    { R"( |   ||   | )" },
    { R"( |   ||   | )" },
    { R"( +===++===+ )" },
    { R"(            )" }};

const std::vector<std::string> openDoor {
    { R"(   ______   )" },
    { R"(|\/      \/|)" },
    { R"(||        ||)" },
    { R"(||        ||)" },
    { R"(||        ||)" },
    { R"(||________||)" },
    { R"(|/        \|)" }};

const std::vector<std::string> sound {
    { R"(    -..-    )" },
    { R"(            )" },
    { R"(  '-,__,-'  )" },
    { R"(            )" },
    { R"( `,_    _,` )" },
    { R"(    `--`    )" },
    { R"(            )" }};

const std::vector<std::string> nosound {
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" }};
// Different format for him!
const std::vector<std::string> man {
    { R"(   ___   )" },
    { R"(  //|\\  )" },
    { R"(  \___/  )" },
    { R"( \__|__/ )" },
    { R"(    |    )" },
    { R"(    |    )" },
    { R"(   / \   )" },
    { R"(  /   \  )" }};

// Random spaces to make the rendering look nice. Yeah this is ugly, but it's
// just for the rendering.
const std::string hspacer{"     "};
const std::string manhspacer(hspacer.size() / 2 + prize[0].size() - man[0].size() / 2, ' ');
const std::string numspacer((prize[0].size() - 8)/2, ' ');

const std::string clockSpacer = numspacer + std::string((hspacer.size() - 1) / 2, ' ');
const std::string strclock(R"(/|\-)");

// MODEL

// enum {
//     A_LISTEN = 0,
//     A_LEFT   = 1,
//     A_MIDDLE = 2,
//     A_RIGHT  = 3,
// };

// enum {
//     TIG_LEFT    = 0,
//     TIG_MIDDLE  = 2,
//     TIG_RIGHT   = 1,
// };

// const int TIG[]

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeTigerProblem(int s,int a,int o) {
    // Actions are: 0-listen, 1-open-left, 2-open-right
    // size_t S = 3, A = 4, O = 3;
    size_t S = s, A = a, O = o;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    for ( size_t a = 0; a < A; ++a ){
        for ( size_t s1 = 0; s1 < S; ++s1 ){
            for ( size_t s2 = 0; s2 < S; ++s2 ){    
                std::cin>>transitions[s1][a][s2];
            }       
        }        
    }
            // std::cout<<"chkpC"<<std::flush;

    for ( size_t a = 0; a < A; ++a )
        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t o = 0; o < O; ++o ){
                 std::cin>>observations[s][a][o];    //final state , action and obs on getting there
        }
    }
            // std::cout<<"chkpC"<<std::flush;

    double rew;
    for ( size_t a = 0; a < A; ++a ){    
        for ( size_t s = 0; s < S; ++s ) {
            std::cin>>rew;
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                rewards[s1][a][s]=rew;
            }            
        }   
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
            // std::cout<<"chkpC"<<std::flush;
    
    model.setObservationFunction(observations);
    return model;
}

int main(int argc, char const *argv[]) {    
    int states,observs,acts,algo,inphorizon, test_horizon,inpnBel,iters,reward_runs;
    double discount;
    std::cin>>algo>>reward_runs>>discount>>inphorizon>>test_horizon>>inpnBel>>iters>>states>>acts>>observs;
    // std::cout<<discount<<std::flush;

    auto model  = makeTigerProblem(states,acts,observs);;
    auto start = std::chrono::high_resolution_clock::now();
    unsigned horizon = inphorizon;        
    
            
    // We create a random engine, since we will need this later.
    // Create model of the problem.
    // std::cout<<"Done1"<<std::flush;
    
    model.setDiscount(discount);
    // Set the horizon. This will determine the optimality of the policy
    // dependent on how many steps of observation/action we plan to do. 1 means
    // we're just going to do one thing only, and we're done. 2 means we get to
    // do a single action, observe the result, and act again. And so on.
    
    AIToolbox::POMDP::Belief b(states);     
    // std::cout<<"Done1"<<std::flush;
        
    for(int belInd=0;belInd<states;belInd++)
    { 
        std::cin>>b(belInd,0);
        std::cout<<"bb"<<b(belInd,0)<<" "<<std::flush;
    }
    std::cout<<"Initial Bel read "<<std::endl;

    if(algo==1)
    {
        AIToolbox::POMDP::IncrementalPruning solver(horizon,0.0);
        auto solution = solver(model);
        // std::cout<<"Done";
        AIToolbox::POMDP::Policy policy(states, acts, observs, std::get<1>(solution));
        
        double totalReward = 0.0;
        for(int runs=0;runs<reward_runs;runs++)
        {
            std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
            auto s = AIToolbox::sampleProbability(states, b, rand);
            auto a_id = policy.sampleAction(b, horizon);
    // std::cout<<"Done1"<<std::flush;

            for (int t = test_horizon - 1; t >= 0; --t) 
            {
    // std::cout<<"Done1"<<std::flush;

                auto currA = std::get<0>(a_id);
                auto s1_o_r = model.sampleSOR(s, currA);
                auto currO = std::get<1>(s1_o_r);
                totalReward += std::get<2>(s1_o_r);
                if (t > (int)policy.getH()-1)
                    a_id = policy.sampleAction(b, policy.getH());
                else
                    a_id = policy.sampleAction(std::get<1>(a_id), currO, t);

                s = std::get<0>(s1_o_r);
            }
        }
        auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
        std::cout<<"Time = "<<training_time<<"\tAverage Total Reward = "<<(totalReward/reward_runs)<<std::endl;
    }
    else if(algo==2)
    {
        AIToolbox::POMDP::POMCP solver(model,inpnBel,iters,10000);
        
        double totalReward = 0.0;        
        for(int runs=0;runs<reward_runs;runs++)
        {
            std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
            auto s = AIToolbox::sampleProbability(states, b, rand);
            auto a_id = solver.sampleAction(b, horizon);

            for (int t = horizon - 1; t >= 0; --t) 
            {
                auto currA = (a_id);
                auto s1_o_r = model.sampleSOR(s, currA);
                auto currO = std::get<1>(s1_o_r);
                totalReward += std::get<2>(s1_o_r);
                a_id = solver.sampleAction(a_id, currO, t);
                s = std::get<0>(s1_o_r);
            }   
        }
        auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
        std::cout<<"Time = "<<training_time<<"\tAverage Total Reward = "<<(totalReward/reward_runs)<<std::endl;
    }
    else
    {
        
        double totalReward = 0.0;        
        
        for(int super_rn =0 ;super_rn<1;super_rn++)
        {
            
            // The 0.0 is the epsilon factor, used with high horizons. It gives a way
            // to stop the computation if the policy has converged to something static.
            // std::cout<<"chkp1"<<std::flush;
        
            
            AIToolbox::POMDP::PolicyIteration solver(horizon,0.0);
            // std::cout<<"chkp2"<<std::flush;

            // Solve the model. After this line, the problem has been completely
            // solved. All that remains is setting up an experiment and see what
            // happens!
            auto solution = std::get<1>(solver(model));

            // for(int mchSt=0;mchSt<(int)solution[0].size();mchSt++)
            // {
            //     std::cout<<"--"<<mchSt<<"--"<<std::endl;
            //     std::cout<<solution[0][mchSt].action<<std::endl;
            //     for(int valI=0;valI<states;valI++)
            //     {
            //         std::cout<<solution[0][mchSt].values[valI]<<" ";
            //     }
            //     std::cout<<std::endl;
            //     for(int valI=0;valI<observs;valI++)
            //     {
            //         std::cout<<solution[0][mchSt].observations[valI]<<" ";
            //     }
            //     std::cout<<std::endl;
            // }

            // std::cout<<"chkp2"<<std::flush;

            // We create a policy from the solution, in order to obtain actual actions
            // depending on what happens in the environment.
            
            // AIToolbox::POMDP::Policy policy(3, 4, 3, std::get<1>(solution));

            for(int runs=0;runs<reward_runs;runs++)
            {
                std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
                auto s = AIToolbox::sampleProbability(states, b, rand);

                // The first thing that happens is that we take an action, so we sample it now.
                double maxExp=solution[0][0].values.dot(b);
                int bAct=solution[0][0].action;
                for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
                {
                    double val = (solution[0][alpInd].values.dot(b)); 
                    if(maxExp < val)
                    {
                        maxExp = val;
                        bAct = (solution[0][alpInd].action);
                    }

                }
                // auto a_id = policy.sampleAction(b, 1);
                // std::cout<<"Initial State "<<s<<" "<<b(s,0)<<std::endl;
                // Setup cout to pretty print the simulation.
                // std::cout.setf(std::ios::fixed, std::ios::floatfield);
                // std::cout.precision(6);

                // We loop for each step we have yet to do.
                // std::cout<<"chkp4"<<std::flush;

                for (int t = test_horizon - 1; t >= 0; --t) {
                    // We get our current action
                    // auto currA = std::get<0>(a_id);
                    auto currA = bAct;
                    // We advance the world one step (the agent only sees the observation
                    // and reward).
                    auto s1_o_r = model.sampleSOR(s, currA);
                    // We get the observation from the model, and update our total reward.
                    auto currO = std::get<1>(s1_o_r);
                    totalReward += std::get<2>(s1_o_r);

                    // { // Rendering of the environment, depends on state, action and observation.
                        // auto & left  = s ? prize : tiger;
                        // auto & right = s ? tiger : prize;
                        // for (size_t i = 0; i < prize.size(); ++i)
                        //     std::cout << left[i] << hspacer << right[i] << '\n';

                        // auto & dleft  = currA == A_LEFT  ? openDoor : closedDoor;
                        // auto & dright = currA == A_RIGHT ? openDoor : closedDoor;
                        // for (size_t i = 0; i < prize.size(); ++i)
                        //     std::cout << dleft[i] << hspacer << dright[i] << '\n';

                        // auto & sleft  = currA == A_LISTEN && currO == TIG_LEFT  ? sound : nosound;
                        // auto & sright = currA == A_LISTEN && currO == TIG_RIGHT ? sound : nosound;
                        // for (size_t i = 0; i < prize.size(); ++i)
                        //     std::cout << sleft[i] << hspacer << sright[i] << '\n';

                        // std::cout << numspacer << b[0] << clockSpacer
                        //           << strclock[t % strclock.size()]
                        //           << clockSpacer << b[1] << '\n';

                        // for (const auto & m : man)
                        //     std::cout << manhspacer << m << '\n';

                        // std::cout << "Timestep missing: " << t << "       \n";
                        // std::cout << "Total reward:     " << totalReward << "       " << std::endl;

                        // goup(3 * prize.size() + man.size() + 3);
                    // }

                    // We explicitly update the belief to show the user what the agent is
                    // thinking. This is also necessary in some cases (depending on
                    // convergence of the solution, see below), otherwise its only for
                    // rendering purpouses. It is a pretty expensive operation so if
                    // performance is required it should be avoided.
                
                    b = AIToolbox::POMDP::updateBelief(model, b, currA, currO);

                    // Now that we have rendered, we can use the observation to find out
                    // what action we should do next.
                    //
                    // Depending on whether the solution converged or not, we have to use
                    // the policy differently. Suppose that we planned for an horizon of 5,
                    // but the solution converged after 3. Then the policy will only be
                    // usable with horizons of 3 or less. For higher horizons, the highest
                    // step of the policy suffices (since it converged), but it will need a
                    // manual belief update to know what to do.
                    //
                    // Otherwise, the policy implicitly tracks the belief via the id it
                    // returned from the last sampling, without the need for a belief
                    // update. This is a consequence of the fact that POMDP policies are
                    // computed from a piecewise linear and convex value function, so
                    // ranges of similar beliefs actually result in needing to do the same
                    // thing (since they are similar enough for the timesteps considered).
                    // if (t > (int)policy.getH())
                    //     a_id = policy.sampleAction(b, policy.getH());
                    // else
                        // a_id = policy.sampleAction(std::get<1>(a_id), currO, t);

                    maxExp=solution[0][0].values.dot(b);
                    bAct=solution[0][0].action;
                    for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
                    {
                        double val = (solution[0][alpInd].values.dot(b)); 
                        if(maxExp < val)
                        {
                            maxExp = val;
                            bAct = (solution[0][alpInd].action);
                        }

                    }
                    // Then we update the world
                    s = std::get<0>(s1_o_r);

                    // std::cout<<"Action: "<<currA<<" Obs: "<<currO<<" State: "<<s<<std::endl;

                //     // Sleep 1 second so the user can see what is happening.
                //     // std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            // std::cout<<"chkp5"<<std::flush;

            // // Put the cursor back where it should be.
            // godown(3 * prize.size() + man.size() + 3);
        }
        auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
        std::cout<<"Time = "<<training_time<<"\tAverage Total Reward = "<<(totalReward/(reward_runs))<<std::endl;
            
    }

    return 0;
}
