// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Neuron.h"

class Neuron;

struct NeuralConnection
{
	double weight;
	double deltaWeight;
};

class NeuralNetwork
{

public:
	NeuralNetwork(const TArray<unsigned> &Topology);
	void FeedForward(const TArray<double> &InputValues);
	void BackwardsPropagation(const TArray<double> &TargetValues);
	void GetResults(TArray<double> &ResultValues) const;

private:
	TArray<TArray<Neuron>> layers;
	double errorDelta;
};
