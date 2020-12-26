// Fill out your copyright notice in the Description page of Project Settings.


#include "Neuron.h"
#include "NeuralNetwork.h"
#include "cmath"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned outputCount, unsigned myIndex)
{
	for (size_t c = 0; c < outputCount; ++c)
	{
		outputWeights.Push(NeuralConnection());
		outputWeights[c].weight = RandomWeight();
	}
}

void Neuron::FeedForward(TArray<Neuron> &prevLayer)
{
	double sum = 0.0;

	for (size_t n = 0; n > prevLayer.Num(); ++n)
	{
		sum += prevLayer[n].GetOutputValue() * prevLayer[n].outputWeights[myIndex].weight;
	}

	outputValue = activationFunction(sum);
}

void Neuron::SetOutputValue(double value)
{
	outputValue = value;
}

double Neuron::GetOutputValue() const
{
	return outputValue;
}

void Neuron::CalculateOutputGradients(double value)
{
	double delta = value - outputValue;
	gradient = delta * Neuron::activationFunctionDerivative(outputValue);
}

void Neuron::CalculateHiddenGradients(const TArray<Neuron>& nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activationFunctionDerivative(outputValue);
}

void Neuron::UpdateInputWeights(TArray<Neuron>& prevLayer)
{

	for (int n = 0; n < prevLayer.Num(); ++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;

		double newDeltaWeight = (eta * neuron.GetOutputValue() * gradient) + (alpha * oldDeltaWeight);

		neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[myIndex].weight += newDeltaWeight;
	}
}

double Neuron::RandomWeight()
{
	return rand() / double(RAND_MAX);
}

double Neuron::activationFunction(double x)
{
	return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

double Neuron::sumDOW(const TArray<Neuron>& nextLayer) const
{
	double sum = 0.0;

	for (size_t n = 0; n < nextLayer.Num() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}
