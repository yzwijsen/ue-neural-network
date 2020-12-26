// Fill out your copyright notice in the Description page of Project Settings.

#include "NeuralNetwork.h"
#include "Neuron.h"
#include "cassert"

NeuralNetwork::NeuralNetwork(const TArray<unsigned> &Topology)
{
	size_t layerCount = Topology.Num();
	for (size_t layerIndex = 0; layerIndex < layerCount; ++layerIndex)
	{
		// Create new layer
		layers.Push(TArray<Neuron>());

		// Set number of outputs so we can feed it into neuron constructor later
		size_t outputCount = layerIndex == Topology.Num() - 1 ? 0 : Topology[layerIndex + 1];
		
		// Fill layer
		for (size_t neuronIndex = 0; neuronIndex <= Topology[layerCount]; ++neuronIndex)
		{
			// Create Neuron
			layers[neuronIndex].Push(Neuron(outputCount, neuronIndex));
		}

		// Set bias node output
		size_t lastLayerIndex = layers.Num() - 1;
		layers[lastLayerIndex][layers[lastLayerIndex].Num() - 1].SetOutputValue(1.0);
	}
}

void NeuralNetwork::FeedForward(const TArray<double> &InputValues)
{
	assert(InputValues.Num() == layers[0].Num() - 1);

	for (size_t i = 0; i < InputValues.Num(); ++i)
	{
		layers[0][i].SetOutputValue(InputValues[i]);
	}

	// Feed Forward
	for (size_t layerIndex = 1; layerIndex < layers.Num(); ++layerIndex)
	{
		TArray<Neuron>& prevLayer = layers[layerIndex - 1];
		for (size_t n = 0; n < layers[layerIndex].Num() - 1; ++n)
		{
			layers[layerIndex][n].FeedForward(prevLayer);
		}
	}
}

void NeuralNetwork::BackwardsPropagation(const TArray<double> &TargetValues)
{
	// Calculate RMS
	TArray<Neuron> &outputLayer = layers[layers.Num() - 1];
	errorDelta = 0.0;

	for (size_t n = 0; n < outputLayer.Num() - 1; ++n)
	{
		double delta = TargetValues[n] - outputLayer[n].GetOutputValue();
		errorDelta += delta * delta;
	}
	errorDelta /= outputLayer.Num() - 1;
	errorDelta = sqrt(errorDelta);

	// Calculate output layer gradients
	for (size_t n = 0; n < outputLayer.Num() - 1; ++n)
	{
		outputLayer[n].CalculateOutputGradients(TargetValues[n]);
	}

	// Calculate gradients on hidden layers
	for (int layerIndex = layers.Num() - 2; layerIndex > 0; --layerIndex)
	{
		TArray<Neuron> &hiddenLayer = layers[layerIndex];
		TArray<Neuron> &nextLayer = layers[layerIndex + 1];

		for (size_t n = 0; n < hiddenLayer.Num(); ++n)
		{
			hiddenLayer[n].CalculateHiddenGradients(nextLayer);
		}
	}

	// Update connection weights
	for (size_t layerIndex = layers.Num() - 1; layerIndex > 0; --layerIndex)
	{
		TArray<Neuron> &layer = layers[layerIndex];
		TArray<Neuron> &prevLayer = layers[layerIndex - 1];

		for (int n = 0; n < layer.Num() - 1; ++n)
		{
			layer[n].UpdateInputWeights(prevLayer);
		}
	}
}

void NeuralNetwork::GetResults(TArray<double> &ResultValues) const
{
	ResultValues.Empty();

	for (size_t n = 0; n < layers[layers.Num() - 1].Num() - 1; ++n)
	{
		ResultValues.Push(layers[layers.Num() - 1][n].GetOutputValue());
	}
}