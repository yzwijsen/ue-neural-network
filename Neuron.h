// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"

struct NeuralConnection;

class Neuron
{
public:
	Neuron(unsigned outputCount, unsigned myIndex);
	void FeedForward(TArray<Neuron> &prevLayer);
	void SetOutputValue(double value);
	double GetOutputValue() const;
	void CalculateOutputGradients(double value);
	void CalculateHiddenGradients(const TArray<Neuron>& nextLayer);
	void UpdateInputWeights(TArray<Neuron>& prevLayer);

private:
	static double RandomWeight();
	double outputValue;
	TArray<NeuralConnection> outputWeights;
	unsigned myIndex;
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);
	double gradient;
	static double alpha;
	static double eta;
	double sumDOW(const TArray<Neuron>& nextLayer) const;
};
