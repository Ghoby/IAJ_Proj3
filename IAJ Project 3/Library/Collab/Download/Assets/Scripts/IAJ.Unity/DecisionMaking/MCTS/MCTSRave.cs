using Assets.Scripts.GameManager;
using System;
using System.Collections.Generic;
using Assets.Scripts.IAJ.Unity.DecisionMaking.GOB;
using UnityEngine;
using Assets.Scripts.IAJ.Unity.Utils;

namespace Assets.Scripts.IAJ.Unity.DecisionMaking.MCTS
{
    public class MCTSRave : MCTS
    {

        protected List<Pair<int, GOB.Action>> ActionHistory { get; set; }

        public MCTSRave(CurrentStateWorldModel currentStateWorldModel) : base(currentStateWorldModel)
        {
        }

        protected override Reward Playout(WorldModel initialPlayoutState)
        {
            //throw new NotImplementedException();
            ActionHistory = new List<Pair<int, GOB.Action>>();
            WorldModel childWorldModel = initialPlayoutState.GenerateChildWorldModel();
            GOB.Action action;

            int playoutReach = 0;

            while (!childWorldModel.IsTerminal())
            {
                //Select a random Action
                action = childWorldModel.GetExecutableActions()[RandomGenerator.Next(0, childWorldModel.GetExecutableActions().Length)];
                ActionHistory.Add(new Pair<int, GOB.Action>(childWorldModel.GetNextPlayer(), action));
                action.ApplyActionEffects(childWorldModel);
                childWorldModel.CalculateNextPlayer();
                playoutReach += 1;
            }

            if (playoutReach > MaxPlayoutDepthReached)
                MaxPlayoutDepthReached = playoutReach;

            Reward reward = new Reward
            {
                PlayerID = childWorldModel.GetNextPlayer(),
                Value = childWorldModel.GetScore()
            };
            return reward;
        }

        protected override void Backpropagate(MCTSNode node, Reward reward)
        {
            while (node != null)
            {
                node.N += 1;
                node.Q += reward.Value;
                if (node.Parent != null)
                    ActionHistory.Add(new Pair<int, GOB.Action>(node.Parent.PlayerID, node.Action));
                node = node.Parent;

                if (node != null)
                {
                    var playerID = node.PlayerID;
                    foreach (var childNode in node.ChildNodes)
                    {

                        if (ActionHistory.Contains(new Pair<int, GOB.Action>(playerID, childNode.Action)))
                        {
                            childNode.NRave += 1;
                            childNode.QRave += childNode.State.GetScore();
                        }
                    }
                }
            }
        }
        protected override MCTSNode BestUCTChild(MCTSNode node)
        {
            MCTSNode bestChildNode = null;
            float bestUCT = -1.0f;
            float RAVE;
            float u;
            float currentUCT;

            //formula for beta brought to you by Wikipedia :) 
            float beta = node.NRave / (node.N + node.NRave + (4 * node.N * node.NRave * 1 * 1)); // b ^ 2 = 1 because b = 1;

            foreach (MCTSNode child in node.ChildNodes)
            {
                RAVE = child.QRave / child.NRave;
                u = child.Q / child.N;
                currentUCT = ((1 - beta) * u + beta * RAVE) + C * (float)Math.Sqrt(Math.Log(child.Parent.N) / child.N);
                if (currentUCT > bestUCT)
                {
                    bestChildNode = child;
                    bestUCT = currentUCT;
                }
            }
            return bestChildNode;
        }

    }
}
