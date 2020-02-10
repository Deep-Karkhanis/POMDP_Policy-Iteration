#ifndef AI_TOOLBOX_POMDP_POLICY_ITERATION_HEADER_FILE
#define AI_TOOLBOX_POMDP_POLICY_ITERATION_HEADER_FILE

#include <limits>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>

#include <Eigen/Dense>
#include <iostream>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <queue>
#include <algorithm>
#include <map>

// #include <map>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the Incremental Pruning algorithm.
     *
     * This algorithm solves a POMDP Model perfectly. It computes solutions
     * for each horizon incrementally, every new solution building upon the
     * previous one.
     *
     * From each solution, it computes the full set of possible
     * projections. It then computes all possible cross-sums of such
     * projections, in order to compute all possible vectors that can be
     * included in the final solution.
     *
     * What makes this method unique is its pruning strategy. Instead of
     * generating every possible vector, combining them and pruning, it
     * tries to prune at every possible occasion in order to minimize the
     * number of possible vectors at any given time. Thus it will prune
     * after creating the projections, after every single cross-sum, and
     * in the end when combining all projections for each action.
     *
     * The performances of this method are *heavily* dependent on the linear
     * programming methods used. In particular, this code currently
     * utilizes the lp_solve55 library. However, this library is not the
     * most efficient implementation, as it defaults to a somewhat slow
     * solver, and its problem-building API also tends to be slow due to
     * lots of bounds checking (which are cool, but sometimes people know
     * what they are doing). Still, to avoid replicating infinite amounts
     * of code and managing memory by ourselves, we use its API. It would
     * be nice if one day we could port directly into the code a fast lp
     * implementation; for now we do what we can.
     */
    class PolicyIteration {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor sets the default horizon used to solve a POMDP::Model.
             *
             * The epsilon parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The epsilon
             * parameter sets the convergence criterion. An epsilon of 0.0
             * forces IncrementalPruning to perform a number of iterations
             * equal to the horizon specified. Otherwise, IncrementalPruning
             * will stop as soon as the difference between two iterations
             * is less than the epsilon specified.
             *
             * @param h The horizon chosen.
             * @param epsilon The epsilon factor to stop the IncrementalPruning loop.
             */
            PolicyIteration(unsigned h, double epsilon, AIToolbox::POMDP::Belief b);

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The epsilon
             * parameter sets the convergence criterion. An epsilon of 0.0
             * forces IncrementalPruning to perform a number of iterations
             * equal to the horizon specified. Otherwise, IncrementalPruning
             * will stop as soon as the difference between two iterations
             * is less than the epsilon specified.
             *
             * @param e The new epsilon parameter.
             */
            void setTolerance(double t);

            /**
             * @brief This function allows setting the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function will return the currently set epsilon parameter.
             *
             * @return The currently set epsilon parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function returns the currently set horizon parameter.
             *
             * @return The current horizon.
             */
            unsigned getHorizon() const;

            /**
             * @brief This function solves a POMDP::Model completely.
             *
             * This function is pretty expensive (as are possibly all POMDP
             * solvers).  It generates for each new solved timestep the
             * whole set of possible ValueFunctions, and prunes it
             * incrementally, trying to reduce as much as possible the
             * linear programming solves required.
             *
             * This function returns a tuple to be consistent with MDP
             * solving methods, but it should always succeed.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction and the computed ValueFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<double, ValueFunction> operator()(const M & model);

        private:
            /**
             * @brief This function computes a VList composed of all possible combinations of sums of the VLists provided.
             *
             * This function performs the job of accumulating the
             * information required to obtain the final policy. Cross-sums
             * are done between each element of each list. The resulting
             * observation output depend on the order parameter, which
             * specifies whether the first list should be put first, or the
             * second one. This parameter is needed due to the order in
             * which we cross-sum all vectors.
             *
             * @param l1 The "main" parent list.
             * @param l2 The list being cross-summed to l1.
             * @param a The action that this cross-sum is about.
             * @param order Which list comes before the other to merge
             *              observations. True for the first, false otherwise.
             *
             * @return The cross-sum between l1 and l2.
             */
            VList crossSum(const VList & l1, const VList & l2, size_t a, bool order);
            
            void updateFSC(const VList &w);
            void makeInitialFSC();
            VList policyEvaluation(const M & model)
   
            size_t S, A, O;
            unsigned horizon_;
            double tolerance_;
            int B_,howard_, nodeLimit_;
            AIToolbox::POMDP::Belief bel_;

            struct MchnState{
                MDP::Values values; //alpha vector is stored here
                size_t action;
                // std::vector<*MchnState> * adjList; 
                VObs obsSucc;
                size_t iden; //identifier
                int mark;

                MchnState()
                {}

                MchnState(MDP::Values v, size_t a, VObs vo, size_t id)
                {
                    values = v;
                    action = a;
                    for(int oi = 0;oi<vo.size();oi++)
                        obsSucc.push_back(vo[oi]);
                    iden = id;
                    mark = 0;
                }
            };

            struct FSC{
                std::vector<MchnState> nodes;
                size_t maxId;
                // std::vector<std::vector<*MchnState> > edges;
                int maxE;
                FSC(){
                  nodes = std::vector<MchnState>(0);
                  maxId = 0;
                  max_E = INT_MIN;
                //   edges = std::vector<std::vector<*MchnState> >(0);
                }
                
            };

            FSC fsc_, fsc_F;


    };



    template <typename M, typename>
    std::tuple<double, ValueFunction> PolicyIteration::operator()(const M & model) {
        // Initialize "global" variables
        S = model.getS();
        A = model.getA();
        O = model.getO();

        makeInitialFSC(); // TODO: May take user input
        unsigned timestep = 0;

        srand(time(NULL));

        Pruner prune(S);
        Projecter projecter(model);

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger
        
        VList alpVecs = policyEvaluation(model);

        while ( timestep<horizon_ && (fsc_F.nodes.size() <= nodeLimit_) && ( !useTolerance || variation > tolerance_ ) )                          
        {
            ++timestep;
            auto projs = projecter(alpVecs);

            size_t finalWSize = 0;
            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t o = 0; o < O; ++o ) {
                    const auto begin = std::begin(projs[a][o]);
                    const auto end   = std::end  (projs[a][o]);
                    projs[a][o].erase(prune(begin, end, unwrap), end);
                }
                bool oddOld = O % 2;
                int i, front = 0, back = O - oddOld, stepsize = 2, diff = 1, elements = O;
                while ( elements > 1 ) {
                    for ( i = front; i != back; i += stepsize ) {
                        projs[a][i] = crossSum(projs[a][i], projs[a][i + diff], a, stepsize > 0);
                        const auto begin = std::begin(projs[a][i]);
                        const auto end   = std::end  (projs[a][i]);
                        projs[a][i].erase(prune(begin, end, unwrap), end);
                        --elements;
                    }

                    const bool oddNew = elements % 2;

                    const int tmp   = back;
                    back      = front - ( oddNew ? 0 : stepsize );
                    front     = tmp   - ( oddOld ? 0 : stepsize );
                    stepsize *= -2;
                    diff     *= -2;

                    oddOld = oddNew;
                }
                if (front != 0)
                    projs[a][0] = std::move(projs[a][front]);
                finalWSize += projs[a][0].size();
            }
            VList w;
            w.reserve(finalWSize);

            for ( size_t a = 0; a < A; ++a )
                 w.insert(std::end(w), std::make_move_iterator(std::begin(projs[a][0])), std::make_move_iterator(std::end(projs[a][0])));

            const auto begin = std::begin(w);
            const auto end   = std::end  (w);
            w.erase(prune(begin, end, unwrap), end);

            if(howard_==0)
            {
                auto fsc_opt = fsc_F;
                auto alpVecs_opt = alpVecs;
                int maxE = INT_MIN;
                for(int branch=0;branch<B_;branch++)
                {
                    VList w1(0);
                    
                    int rnd_numb;
                    int vecsChosen=0;
                    while(vecsChosen == 0)
                    {
                        vecsChosen=0;
                        for(size_t alpDInd=0;alpDInd<(w.size()); alpDInd++)
                        {
                            rnd_numb = rand() % 2;
                            if(rnd_numb == 1)
                            {
                                auto alpD = w[alpDInd];
                                for(int obsInd=0;obsInd<O;obsInd++)
                                {
                                    w1.push_back(alpD);
                                }
                                vecsChosen++;
                            }
                        }
                    }      
                    updateFSC(w1);
                    alpVecs = policyEvaluation(model);
            
                    if(maxE < fsc_.maxE)
                    {
                        fsc_opt = fsc_;    
                        alpVecs_opt = alpVecs;           
                    }
                    else if(maxE == fsc_.maxE)
                    {
                        rnd_numb = rand() % 2;
                        if(rnd_numb == 1)
                        {
                            fsc_opt = fsc_;    
                            alpVecs_opt = alpVecs;           
                        }
                    }
                }
                fsc_F = fsc_opt;
                alpVecs = alpVecs_opt;
            }
            else
            {
                updateFSC(w);
                alpVecs = policyEvaluation(model);
                fsc_F = fsc_;
            }

        }

        ValueFunction v(0);
        VList vl(0);
        for(int i=0;i<(int)(fsc_.nodes.size());i++)
        {
            VEntry ve(fsc_.nodes[i].values,fsc_.nodes[i].action,fsc_.nodes[i].obsSucc);
            vl.push_back(ve);
        }
        v.push_back(vl);
        std::cout<<"FSC Size "<<fsc_.nodes.size()<<std::endl<<std::flush;
        return std::make_tuple(useTolerance ? variation : 0.0, v);
    }
}

#endif
