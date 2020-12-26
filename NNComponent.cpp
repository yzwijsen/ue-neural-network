// Fill out your copyright notice in the Description page of Project Settings.


#include "NNComponent.h"
#include "NeuralNetwork.h"

// Sets default values for this component's properties
UNNComponent::UNNComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UNNComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	TArray<unsigned> Topology;
	Topology.Push(3);
	Topology.Push(2);
	Topology.Push(1);

	NeuralNetwork NeuralNet(Topology);

	TArray<double> InputValues;
	NeuralNet.FeedForward(InputValues);

	TArray<double> TargetValues;
	NeuralNet.BackwardsPropagation(TargetValues);

	TArray<double> ResultValues;
	NeuralNet.GetResults(ResultValues);
	
}


// Called every frame
void UNNComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

