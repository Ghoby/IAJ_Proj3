﻿using System;
using System.Collections.Generic;
using RAIN.Navigation.Graph;

namespace Assets.Scripts.IAJ.Unity.Pathfinding.DataStructures
{
    public class NodeRecordArray : IOpenSet, IClosedSet
    {
        private NodeRecord[] NodeRecords { get; set; }
        private List<NodeRecord> SpecialCaseNodes { get; set; } 
        private NodePriorityHeap Open { get; set; }

        public NodeRecordArray(List<NavigationGraphNode> nodes)
        {
            //this method creates and initializes the NodeRecordArray for all nodes in the Navigation Graph
            this.NodeRecords = new NodeRecord[nodes.Count];
            
            for(int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                node.NodeIndex = i; //we're setting the node Index because RAIN does not do this automatically
                this.NodeRecords[i] = new NodeRecord {node = node, status = NodeStatus.Unvisited};
            }

            this.SpecialCaseNodes = new List<NodeRecord>();

            this.Open = new NodePriorityHeap();
        }

        public NodeRecord GetNodeRecord(NavigationGraphNode node)
        {
            //do not change this method
            //here we have the "special case" node handling
            if (node.NodeIndex == -1)
            {
                for (int i = 0; i < this.SpecialCaseNodes.Count; i++)
                {
                    if (node == this.SpecialCaseNodes[i].node)
                    {
                        return this.SpecialCaseNodes[i];
                    }
                }
                return null;
            }
            else
            {
                return  this.NodeRecords[node.NodeIndex];
            }
        }

        public void AddSpecialCaseNode(NodeRecord node)
        {
            this.SpecialCaseNodes.Add(node);
        }

        void IOpenSet.Initialize()
        {
            this.Open.Initialize();
            //we want this to be very efficient (that's why we use for)
            for (int i = 0; i < this.NodeRecords.Length; i++)
            {
                this.NodeRecords[i].status = NodeStatus.Unvisited;
            }

            this.SpecialCaseNodes.Clear();
        }

        void IClosedSet.Initialize()
        {
            for (int i = 0; i < this.NodeRecords.Length; i++)
            {
                this.NodeRecords[i].status = NodeStatus.Unvisited;
            }

            this.SpecialCaseNodes.Clear();
        }

        public void AddToOpen(NodeRecord nodeRecord)
        {
			this.GetNodeRecord(nodeRecord.node).status = NodeStatus.Open;
            this.GetNodeRecord(nodeRecord.node).parent = nodeRecord.parent;
            this.GetNodeRecord(nodeRecord.node).gValue = nodeRecord.gValue;
            this.GetNodeRecord(nodeRecord.node).hValue = nodeRecord.hValue;
            this.GetNodeRecord(nodeRecord.node).fValue = nodeRecord.fValue;
			

			this.Open.AddToOpen(this.GetNodeRecord(nodeRecord.node));
        }

        public void AddToClosed(NodeRecord nodeRecord) 
        {
			this.GetNodeRecord(nodeRecord.node).status = NodeStatus.Closed;
		}

        public NodeRecord SearchInOpen(NodeRecord nodeRecord)
        {
			if (this.GetNodeRecord(nodeRecord.node).status == NodeStatus.Open)
				return this.GetNodeRecord(nodeRecord.node);
			else
				return null;

		}

        public NodeRecord SearchInClosed(NodeRecord nodeRecord)
        {
			if (this.GetNodeRecord(nodeRecord.node).status == NodeStatus.Closed)
				return this.GetNodeRecord(nodeRecord.node);
			else
				return null;

		}

        public NodeRecord GetBestAndRemove()
        {
            var best = this.PeekBest();
            this.RemoveFromOpen(best);

            return best;
        }

        public NodeRecord PeekBest()
        {
            return this.Open.PeekBest(); 
        }

        public void Replace(NodeRecord nodeToBeReplaced, NodeRecord nodeToReplace)
        {
			/*this.Open.Replace(nodeToBeReplaced, nodeToReplace);
            nodeToBeReplaced.parent = nodeToReplace.parent;
            nodeToBeReplaced.gValue = nodeToReplace.gValue;
            nodeToBeReplaced.hValue = nodeToReplace.hValue;
            nodeToBeReplaced.fValue = nodeToReplace.fValue;*/

			this.Open.RemoveFromOpen(nodeToBeReplaced);
			this.AddToOpen(nodeToReplace);


        }

        public void RemoveFromOpen(NodeRecord nodeRecord)
        {
			this.Open.RemoveFromOpen(nodeRecord);
			this.GetNodeRecord(nodeRecord.node).status = NodeStatus.Visited;

		}

        public void RemoveFromClosed(NodeRecord nodeRecord)
        {
			this.GetNodeRecord(nodeRecord.node).status = NodeStatus.Visited;
		}

        ICollection<NodeRecord> IOpenSet.All()
        {
            return this.Open.All();
        }

        ICollection<NodeRecord> IClosedSet.All()
        {
            List<NodeRecord> aux = new List<NodeRecord>();
            for (int i = 0; i < this.NodeRecords.Length; i++)
            {
                if (this.NodeRecords[i].status == NodeStatus.Closed)
                {
                    aux.Add(this.NodeRecords[i]);
                }
            }
            return aux;
        }

        public int CountOpen()
        {
            return this.Open.CountOpen();
        }
    }
}
