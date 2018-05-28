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
            ActionHistory = new List<Pair<int, GOB.Action>>();
            WorldModel childWorldModel = initialPlayoutState.GenerateChildWorldModel();
            GOB.Action[] actions = childWorldModel.GetExecutableActions();
            while (!childWorldModel.IsTerminal())
            {
                if (actions.Length > 0)
                {
                    int index = this.RandomGenerator.Next(actions.Length);
                    GOB.Action a = actions[index];
                    var playerActionPair = new Pair<int, GOB.Action>(childWorldModel.GetNextPlayer(), a);
                    ActionHistory.Add(playerActionPair);
                    a.ApplyActionEffects(childWorldModel);
                    childWorldModel.CalculateNextPlayer();
                }
            }

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
                if (reward.PlayerID == node.Parent.PlayerID)
                {
                    node.Q += reward.Value;
                }
                else
                {
                    node.Q -= reward.Value;
                }
                var playerActionPair = new Pair<int, GOB.Action>(node.Parent.PlayerID, node.Action);
                ActionHistory.Add(playerActionPair);
                node = node.Parent;

                if (node != null)
                {
                    int playerID = node.PlayerID;
                    
                    foreach(var childNode in node.ChildNodes)
                    {
                        var playerActionPair2 = new Pair<int, GOB.Action>(node.Parent.PlayerID, node.Action);
                        if (ActionHistory.Contains(playerActionPair2))
                        {
                            node.NRave += 1;
                            node.QRave += reward.Value;
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

            //formula for beta brought to you by Wikipedia :) 
            var beta = node.NRave / (node.N + node.NRave + (4 * node.N * node.NRave * 1 * 1)); // b ^ 2 = 1 because b = 1;

            foreach (var child in node.ChildNodes)
            {
                RAVE = child.NRave / child.QRave;
                var u = child.Q / child.N;

                var currentUCT = ((1 - beta) * u + beta * RAVE) + C * Mathf.Sqrt(Mathf.Log(child.Parent.N) / child.N);

                if (currentUCT > bestUCT)
                {
                    bestUCT = currentUCT;
                    bestChildNode = child;
                }
            }
            return bestChildNode;
        }
    }
}
