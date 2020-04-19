"Evaluate before training"
    pre_train_evaluator = ModelEvaluator(np.arange(0, 10), "PRE-TRAIN")
    for input, label in zip(test_images, test_labels):
        output = net.predict(mnist_input_as_one_d_array(input))
        label_vector = mnist_label_as_one_hot(label)
        loss = loss_func.loss(output, label_vector)
        predicted = highest_output_neuron(output)
        pre_train_evaluator.receive(predicted, int(label), loss)
    
    "Train"
    train(
        net,
        train_images,
        train_labels,
        mnist_input_as_one_d_array,
        mnist_label_as_one_hot,
        epoch_count=10000
    )

    "Evaluate after training"
    post_train_evaluator = ModelEvaluator(np.arange(0, 10), "POST-TRAIN")
    for input, label in zip(test_images, test_labels):
        output = net.predict(mnist_input_as_one_d_array(input))
        label_vector = mnist_label_as_one_hot(label)
        loss = loss_func.loss(output, label_vector)
        predicted = highest_output_neuron(output)
        post_train_evaluator.receive(predicted, int(label), loss)
        
    pre_train_evaluator.plot()
    post_train_evaluator.plot()