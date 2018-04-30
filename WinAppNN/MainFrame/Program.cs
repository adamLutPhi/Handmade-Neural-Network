using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WinAppNN;

namespace MainFrame
{
    class Program
    {
       static int epochs = 5000;
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome!\nour Neural Network is now to Study the 3-Input XOR Pattern!");
            Console.WriteLine("Welcome!\nwe first start by training the Neural Network for you!");
            NeuralNetwork nn = new NeuralNetwork(new int[] { 3,25,25,1});
         //Neural Network Training!
            for (int i = 0; i < epochs; i++)
            {
                nn.FeedForward(new float[] { 0,0,0}); //INPUT LAYER of 3 Neurons
                nn.BackProp(new float[]  { 0 }); //HAS THIS Expected Output Layer of 1 Neuron

                nn.FeedForward(new float[] { 0,0,1});
                nn.BackProp(new float[] { 1 });

                nn.FeedForward(new float[] { 0, 1, 0 });
                nn.BackProp(new float[] { 1 });

                nn.FeedForward(new float[] { 0, 1, 1});
                nn.BackProp(new float[] { 0 });

                nn.FeedForward(new float[] { 1, 0, 0 });
                nn.BackProp(new float[] { 1 });

                nn.FeedForward(new float[] { 1, 0, 1 });
                nn.BackProp(new float[] { 0 });

                nn.FeedForward(new float[] { 1, 1, 0 });
                nn.BackProp(new float[] { 0 });

                nn.FeedForward(new float[] { 1, 1, 1 });
                nn.BackProp(new float[] { 1 });
            }

            //Finished Training
            
            Console.WriteLine("Finished Training!!!");
            Console.WriteLine("Testing Area:\n");
            var o=  nn.FeedForward(new float[] { 0,0,0});
            
            Console.WriteLine("the Output for  Sequence 0 0 0 is: "+o[0]);
            o = nn.FeedForward(new float[] { 0, 0, 1 });
            Console.WriteLine("the Output for  Sequence 0 0 1 is: " + o[0]);

            o = nn.FeedForward(new float[] { 1, 0, 1 });
            Console.WriteLine("the Output for  Sequence 1 0 1 is: " + o[0]);

            o = nn.FeedForward(new float[] { 1, 1, 1 });
            Console.WriteLine("the Output for  Sequence 1 1 1 is: " + o[0]);

            Console.WriteLine("Done! Press Any Key to Close...");
            Console.ReadKey();
        }
    }
}
