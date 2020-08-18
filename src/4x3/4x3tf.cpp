#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>
#include <chrono>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <Eigen/Core>
#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/PolicyIteration.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Algorithms/PBPI.hpp>
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include <AIToolbox/POMDP/Algorithms/PIMCP.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/Impl/Seeder.hpp>



void test_algorithm(const AIToolbox::POMDP::Model<AIToolbox::MDP::Model> &model, const AIToolbox::POMDP::ValueFunction &solution, int reward_runs, int states, double discount, AIToolbox::POMDP::Belief &b, int test_horizon, std::chrono::time_point<std::chrono::high_resolution_clock> &start, std::default_random_engine &myrand){
    double totalReward = 0.0;
    AIToolbox::POMDP::Belief b_orig = b;
    // for(int belInd=0;belInd<states;belInd++)
    // {
    //     std::cin>>b_orig(belInd,0);
    //     std::cout<<"bb"<<b_orig(belInd,0)<<" "<<std::flush;
    // }
    // std::cout<<"chkp2"<<std::flush;
    // AIToolbox::POMDP::Policy policy(3, 4, 3, std::get<1>(solution));

    for(int runs=0;runs<reward_runs;runs++)
    {
        double gamma_pow = 1.0;
        b = b_orig;

        auto s = AIToolbox::sampleProbability(states, b, myrand);

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
            totalReward += gamma_pow*std::get<2>(s1_o_r);
            gamma_pow = gamma_pow*discount;

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
    double maxExp=solution[0][0].values.dot(b_orig);
    for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
    {
        double val = (solution[0][alpInd].values.dot(b_orig));
        if(maxExp < val)
            maxExp = val;
    }

    auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    std::cout<<"Time = "<<training_time<<"\tAverage Total (Discounted) Reward = "<<(totalReward/(reward_runs))<<std::endl;
    std::cout<<"Maximum Expected Reward = "<<maxExp<<std::endl;
    return;
}


inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeProblem(int s,int a,int o, int tiger) {
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
    if(tiger == 1)
        for ( size_t a = 0; a < A; ++a ){
            for ( size_t s = 0; s < S; ++s ) {
                std::cin>>rew;
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    rewards[s][a][s1]=rew;
                }
            }
        }
    else
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
    unsigned my_seed = 42;
    unsigned tiger = 0;
    for(int i = 1; i<argc; i++){
        if(argv[i] == std::string("-tiger")) tiger = 1;
        else if(argv[i] == std::string("-seed")) my_seed = std::atoi(argv[++i]);
    }
    std::cout<<"seed: "<<my_seed<<"\n";
    std::default_random_engine myrand(my_seed);
    AIToolbox::Impl::Seeder::setRootSeed(my_seed);
    int states,observs,acts,algo,inphorizon, test_horizon,inpnBel,iters,reward_runs;

    double discount;
    std::cin>>algo
            >>reward_runs
            // number of times to reapeat the experiment
            >>discount
            >>inphorizon>>test_horizon
            // inphorizon is the test_horizon for algo = 3.
            >>inpnBel
            // don't know
            >>iters
            // Max number of planning iterations
            >>states>>acts>>observs;

    // std::cout<<discount<<std::flush;

    // system("mkdir dummy");
    // exit(0);

    auto model  = makeProblem(states,acts,observs, tiger);
    auto start = std::chrono::high_resolution_clock::now();
    unsigned horizon = inphorizon;

    // create a random engine, since we will need this later.
    // Create model of the problem.
    // std::cout<<"Done1"<<std::flush;
    model.setDiscount(discount);

    AIToolbox::POMDP::Belief b(states);
    // std::cout<<"belief\n"<<std::flush;
    for(int belInd=0;belInd<states;belInd++)
    {
        std::cin>>b(belInd,0);
        std::cout<<"bb"<<b(belInd,0)<<" "<<std::flush;
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
            // std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
            auto s = AIToolbox::sampleProbability(states, b, myrand);
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
        // std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());

        double totalReward = 0.0;
        for(int runs=0;runs<reward_runs;runs++)
        {

            auto s = AIToolbox::sampleProbability(states, b, myrand);
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
        std::cout<<soln.size()<<"\n";
        std::cout<<soln[0].size()<<"\n";
        std::cout<<soln[soln.size()-1].size()<<"\n";
        AIToolbox::POMDP::PIMCP solver(model,inpnBel,iters,10000,soln);

        double totalReward = 0.0;
        for(int runs=0;runs<reward_runs;runs++)
        {
            // std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
            auto s = AIToolbox::sampleProbability(states, b, myrand);
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
    else if(algo == 4){
        AIToolbox::POMDP::PBVI solver1(100, 1, 0.0);
        AIToolbox::POMDP::ValueFunction v = AIToolbox::POMDP::makeValueFunction(states);
        // auto soln = std::get<1>(solver1(model));
        std::vector<AIToolbox::POMDP::Belief> bList;
        bList.emplace_back(b);
        int iter = 0;
        while(iter < iters)
        {
            iter++;
            v = std::get<1>(solver1(model, bList, v));
            std::cout<<v.size()<<"\n";
            std::cout<<v[0].size()<<"\n";

            size_t belief_points = bList.size();
            std::cout<<v[v.size()-1].size()<<", "<<belief_points<<"\n";
            for(int i=0; i<belief_points; i++)
            {
                std::vector<AIToolbox::POMDP::Belief> new_beliefs;
                double max_L1_dist = 0; //have to set it to epsilon later
                size_t max_b_ind = 0;
                for(int act=0; act<acts; act++)
                {
                    AIToolbox::ProbabilityVector z_prob(observs);
                    for(int ob=0; ob<observs; ob++)
                        z_prob[ob] = updateBeliefUnnormalized(model, bList[i], act, ob).sum();
                    // std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
                    auto z = AIToolbox::sampleProbability(observs, z_prob, myrand);
                    new_beliefs.emplace_back(updateBelief(model, bList[i], act, z));
                    double l1disttoB = (bList[0]-new_beliefs.back()).lpNorm<1>();
                    for(int ind=1; ind<belief_points; ind++)
                    {
                        double norm = (bList[ind]-new_beliefs.back()).lpNorm<1>();
                        if(norm < l1disttoB) l1disttoB = norm;
                    }
                    if(l1disttoB >= max_L1_dist)
                    {
                        max_L1_dist = l1disttoB;
                        max_b_ind = act;
                    }
                }
                bList.emplace_back(new_beliefs[max_b_ind]);
                // bList[i]
            }
        }


        AIToolbox::POMDP::PIMCP solver(model,inpnBel,iters,10000,v);
        double totalReward = 0.0;
        for(int runs=0;runs<reward_runs;runs++)
        {
            // std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
            auto s = AIToolbox::sampleProbability(states, b, myrand);
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
    else if(algo == 5)
    {
        double eps = 0.0;
        AIToolbox::POMDP::PBPI solver1(100, iters, 0.0, eps, b);
        auto soln = std::get<1>(solver1(model, myrand));
        std::cout<<soln[0].size()<<"\n";
        test_algorithm(model, soln, reward_runs, states, discount, b, horizon, start, myrand);
        // std::cout<<soln[0].size()<<"\n";
        // std::cout<<soln[soln.size()-1].size()<<"\n";
        // AIToolbox::POMDP::PIMCP solver(model,inpnBel,iters,10000,soln);
        //
        // double totalReward = 0.0;
        // for(int runs=0;runs<reward_runs;runs++)
        // {
        //     std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
        //     auto s = AIToolbox::sampleProbability(states, b, rand);
        //     auto a_id = solver.sampleAction(b, horizon);
        //
        //     for (int t = horizon - 1; t >= 0; --t)
        //     {
        //         auto currA = (a_id);
        //         auto s1_o_r = model.sampleSOR(s, currA);
        //         auto currO = std::get<1>(s1_o_r);
        //         totalReward += std::get<2>(s1_o_r);
        //         a_id = solver.sampleAction(a_id, currO, t);
        //         s = std::get<0>(s1_o_r);
        //     }
        // }
        // auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
        // std::cout<<"Time = "<<training_time<<"\tAverage Total Reward = "<<(totalReward/reward_runs)<<std::endl;
    }
    else
    {
        for(int super_rn =0 ;super_rn<1;super_rn++)
        {
            double totalReward = 0.0;
            AIToolbox::POMDP::PolicyIteration solver(iters,0.0,b);
            // std::cout<<"chkp2"<<std::flush;

            auto solution = std::get<1>(solver(model));
            test_algorithm(model, solution, reward_runs, states, discount, b, test_horizon, start, myrand);
            // AIToolbox::POMDP::Policy policy(3, 4, 3, std::get<1>(solution));

            // for(int runs=0;runs<reward_runs;runs++)
            // {
            //     std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
            //     auto s = AIToolbox::sampleProbability(states, b, rand);
            //
            //     double maxExp=solution[0][0].values.dot(b);
            //     int bAct=solution[0][0].action;
            //     for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
            //     {
            //         double val = (solution[0][alpInd].values.dot(b));
            //         if(maxExp < val)
            //         {
            //             maxExp = val;
            //             bAct = (solution[0][alpInd].action);
            //         }
            //     }
            //
            //     // std::cout<<"chkp4"<<std::flush;
            //
            //     for (int t = test_horizon - 1; t >= 0; --t) {
            //         auto currA = bAct;
            //         // We advance the world one step (the agent only sees the observation
            //         // and reward).
            //         auto s1_o_r = model.sampleSOR(s, currA);
            //         // We get the observation from the model, and update our total reward.
            //         auto currO = std::get<1>(s1_o_r);
            //         totalReward += std::get<2>(s1_o_r);
            //
            //         b = AIToolbox::POMDP::updateBelief(model, b, currA, currO);
            //
            //         // Now that we have rendered, we can use the observation to find out
            //         // what action we should do next.
            //         //
            //         // Depending on whether the solution converged or not, we have to use
            //         // the policy differently. Suppose that we planned for an horizon of 5,
            //         // but the solution converged after 3. Then the policy will only be
            //         // usable with horizons of 3 or less. For higher horizons, the highest
            //         // step of the policy suffices (since it converged), but it will need a
            //         // manual belief update to know what to do.
            //         //
            //         // Otherwise, the policy implicitly tracks the belief via the id it
            //         // returned from the last sampling, without the need for a belief
            //         // update. This is a consequence of the fact that POMDP policies are
            //         // computed from a piecewise linear and convex value function, so
            //         // ranges of similar beliefs actually result in needing to do the same
            //         // thing (since they are similar enough for the timesteps considered).
            //         // if (t > (int)policy.getH())
            //         //     a_id = policy.sampleAction(b, policy.getH());
            //         // else
            //         //     a_id = policy.sampleAction(std::get<1>(a_id), currO, t);
            //
            //         maxExp=solution[0][0].values.dot(b);
            //         bAct=solution[0][0].action;
            //         for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
            //         {
            //             double val = (solution[0][alpInd].values.dot(b));
            //             if(maxExp < val)
            //             {
            //                 maxExp = val;
            //                 bAct = (solution[0][alpInd].action);
            //             }
            //
            //         }
            //         // Then we update the world
            //         s = std::get<0>(s1_o_r);
            //
            //         // std::cout<<"Action: "<<currA<<" Obs: "<<currO<<" State: "<<s<<std::endl;
            //     }
            // }
            // // std::cout<<"chkp5"<<std::flush;
            //
            // double maxExp=solution[0][0].values.dot(b);
            // for(int alpInd=0;alpInd<(int)(solution[0].size());alpInd++)
            // {
            //     double val = (solution[0][alpInd].values.dot(b));
            //     if(maxExp < val)
            //         maxExp = val;
            // }
            //
            // auto training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
            // std::cout<<"Time = "<<training_time<<"\tAverage Total Reward = "<<(totalReward/(reward_runs))<<std::endl;
            // std::cout<<"Expected Reward = "<<maxExp<<std::endl;
        }


    }
    return 0;
}
