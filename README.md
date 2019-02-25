# Episodic_Training_Domain_generalization
My implantation of paper "Episodic Training for Domain Generalization"

# Run
run python file:
python3 torch_train.py

# Note!
There may be a minus different between my implantation and the original paper, when computing the loss of $\phi$. 
I didn't find any information about how to share a loss between two different optimizers. 
So I just update the parameter $\theata$ and recompute the loss $L_{agg}$ using the updated $\theta$ and then update $\phi$.
If anyone know a better method to do this, please tell me, I would be very grateful.
