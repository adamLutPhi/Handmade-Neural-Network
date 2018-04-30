using System;

namespace WinAppNN
{
    public class Layer
    {
       public  int numberofInputs; // number of Neurons of the previous layer
       public int numberofOutputs; // number of Neurons of the Current layer

       public  float[] outputs;
       public float[] inputs;

       public  float[,] weights;
       public float[,] weightsDelta;
       public float[] gamma;
       public float[] error;
        public float LearningRate = 0.01f;

        public enum CostFunctionType { MSE,SSE}

        public static Random MyRnd = new Random();

        public Layer(int numberofInputs, int numberofOutputs)
        {
            this.numberofInputs = numberofInputs;
            this.numberofOutputs = numberofOutputs;

            inputs = new float[numberofInputs];
            outputs = new float[numberofOutputs];
            weights = new float[numberofOutputs, numberofInputs];
            weightsDelta = new float[numberofOutputs, numberofInputs];

            gamma = new float[numberofOutputs]; 
            error = new float[numberofOutputs];

            //Don't forget to Initialize Weights !!
            InitalizeWeights();
        }
        public void InitalizeWeights()
        {
            for (int i = 0; i < numberofOutputs; i++)
            {
                for (int j = 0; j < numberofInputs; j++)
                {
                    weights[i, j] = (float)MyRnd.NextDouble() - 0.5f;
                }
            }
        }
        public void UpdateWeights()
        {
            //subtract Delta values from  our Weights
            for (int i = 0; i < numberofOutputs; i++)
            {
                for (int j = 0; j < numberofInputs; j++)
                {
                    weights[i, j] -= weightsDelta[i, j]*LearningRate;
                }
            }
        }

        public float TanhDer(float value)
        {
          return  1 - (value * value);
        }
        public float CalcCostFunction(float[] expected,CostFunctionType cType)
        {
            for (int i = 0; i < numberofOutputs; i++)
            {
                error[i] = outputs[i] - expected[i];
            }
            float sol = 0.0f;
            switch (cType)
            {
                case CostFunctionType.MSE:
                    int cnt = 0; float sum = 0;
                    for (int i = 0; i < error.Length; i++)
                    {
                        sum += error[i];
                        cnt++;
                    }
                    sol= sum / cnt;
                    break;
                case CostFunctionType.SSE:
                      sum = 0;
                    for (int i = 0; i < error.Length; i++)
                    {
                        sum += error[i];
                    }
                    sol = sum;
                    break;
                default:
                    break;
            }
            return sol;
        }
        public void BackPropOutput(float[] expected) //for  the output -last Layer!
        {
            //calculate the error Function!
            for (int i = 0; i < numberofOutputs; i++)
            {
                error[i] = outputs[i] - expected[i];
            }
            for (int i = 0; i < numberofOutputs; i++)
            {
                gamma[i] = error[i] * TanhDer(outputs[i]);
            }
            for (int i = 0; i < numberofOutputs; i++)
            {
                for (int j = 0; j < numberofInputs; j++)
                {
                    weightsDelta[i,j]= gamma[i] * inputs[j];
                }
            }
        }

        public void BackPropHidden(float[] gammaForward, float[,] WeightsForward)
        {
            for (int i = 0; i < numberofOutputs; i++)
            {
                gamma[i] = 0; //zero out the gamma value!

                for (int j = 0; j < gammaForward.Length; j++)
                {
                    gamma[i] += gammaForward[j]* WeightsForward[j,i];
                }
                     gamma[i] *= TanhDer(outputs[i]);
            }

            for (int i = 0; i < numberofOutputs; i++)
            {
                for (int j = 0; j < numberofInputs; j++)
                {
                    weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }

        }

        public float[] FeedForward(float[] inputs) //feedforward
        {
            
            this.inputs = inputs;
            for (int i = 0; i < numberofOutputs; i++)
            {
                outputs[i] = 0;
               
                for (int j = 0; j < numberofInputs; j++)
                {
                    outputs[i] += inputs[j] * weights[i,j];
                }
                outputs[i] = (float)Math.Tanh(outputs[i]);
            }

            return outputs;
        }

        enum Activationfunctions { tanh,Step, UnImplemented }

        public void ManageAFs(int num)
        {
            Activationfunctions av = (Activationfunctions)Enum.ToObject(typeof(Activationfunctions), num);

            switch (av)
            {
                case Activationfunctions.tanh:
                    {

                        break;
                    }
                case Activationfunctions.Step:
                    {

                        break;
                    }
                case Activationfunctions.UnImplemented:
                    {

                        break;
                    }
                default:
                    {
                        break;
                    }

            }
        }
    }
}