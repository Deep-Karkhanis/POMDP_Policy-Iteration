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
#include <AIToolbox/POMDP/Algorithms/PIMCP.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeProblem(int s,int a,int o) {
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
    for ( size_t s = 0; s < S; ++s ) {
        std::cin>>rew;
        for ( size_t a = 0; a < A; ++a ){    
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
    // std::cin>>algo
            // >>reward_runs
            // >>discount
            // >>inphorizon>>test_horizon
            // >>inpnBel
            // >>iters
            // >>states>>acts>>observs;
    // std::cout<<discount<<std::flush;

    system("mkdir dummy");
    exit(0);

    auto model  = makeProblem(states,acts,observs);;
    auto start = std::chrono::high_resolution_clock::now();
    unsigned horizon = inphorizon;        
    
    // create a random engine, since we will need this later.
    // Create model of the problem.
    // std::cout<<"Done1"<<std::flush;
    model.setDiscount(discount);
    
    AIToolbox::POMDP::Belief b(states);     
    // std::cout<<"Done1"<<std::flush;    
    for(int belInd=0;belInd<states;belInd++)
    { 
        std::cin>>b(belInd,0);
        // std::cout<<"bb"<<b(belInd,0)<<" "<<std::flush;
    }
    // std::cout<<"Initial Bel read "<<std::endl;

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
    else if(algo==3)
    {
        AIToolbox::POMDP::PolicyIteration solver1(iters,0.0,b);
        auto soln = std::get<1>(solver1(model));
        AIToolbox::POMDP::PIMCP solver(model,inpnBel,iters,10000,soln);

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
        for(int super_rn =0 ;super_rn<1;super_rn++)
        {
            double totalReward = 0.0;               
            AIToolbox::POMDP::PolicyIteration solver(iters,0.0,b);
            // std::cout<<"chkp2"<<std::flush;

            auto solution = std::get<1>(solver(model));
            // AIToolbox::POMDP::Policy policy(3, 4, 3, std::get<1>(solution));

            for(int runs=0;runs<reward_runs;runs++)
            {
                std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
                auto s = AIToolbox::sampleProbability(states, b, rand);

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
                
                // std::cout<<"chkp4"<<std::flush;

                for (int t = test_horizon - 1; t >= 0; --t) {
                    auto currA = bAct;
                    // We advance the world one step (the agent only sees the observation
                    // and reward).
                    auto s1_o_r = model.sampleSOR(s, currA);
                    // We get the observation from the model, and update our total reward.
                    auto currO = std::get<1>(s1_o_r);
                    totalReward += std::get<2>(s1_o_r);

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
                    //     a_id = policy.sampleAction(std::get<1>(a_id), currO, t);

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
                }
            }
            // std::cout<<"chkp5"<<std::flush;
            
            double maxExp=solution[0][0].values.dot(b);
            for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
            {
                double val = (solution[0][alpInd].values.dot(b)); 
                if(maxExp < val)
                    maxExp = val;
            }
                    
            auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
            std::cout<<"Time = "<<training_time<<"\tAverage Total Reward = "<<(totalReward/(reward_runs))<<std::endl;
            std::cout<<"Expected Reward = "<<maxExp<<std::endl;
        }

        
    }
    return 0;
}