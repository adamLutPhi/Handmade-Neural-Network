using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WinAppNN
{
    public class NeuralNetwork
    {
        int[] layer;
        Layer[] layers;

        public NeuralNetwork(int[] layer)
        {
            this.layer = new int[layer.Length];
            for (int i = 0; i < layer.Length; i++)
            {
                this.layer[i] = layer[i];
            }

            layers = new Layer[layer.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(layer[i], layer[i + 1]);
                //TODO:Initialize 2nd for loop to init every Neuron.class
            }
        }

        public float[] FeedForward(float[] inputs)
        {
            layers[0].FeedForward(inputs);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward(layers[i-1].outputs);
            }
            return layers[layers.Length - 1].outputs;
        }

        public void BackProp(float[] expected)
        {

            //start back propagating from the last layer!

            // for layers.Length - 1:

            layers[layers.Length - 1].BackPropOutput(expected);

            for (int i = layers.Length-2; i>=0; i--)
            {
                //if (i == layers.Length - 1)
                //{
                //    layers[i].BackPropOutput(expected);
                //}
                //else
                //{
                    layers[i].BackPropHidden(layers[i+1].gamma,layers[i+1].weights);
               // }

            }

            //Iterate over all the Layers - then  Update the weights
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].UpdateWeights();
            }
        }
    }
}
