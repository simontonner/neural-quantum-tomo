- [ ] The dataloader doesn't really respect the seed. Only the model is truly reseeded.
- [ ] However, to truly test data bias we should even generate the data with a different seed. Well at least the
        sampling not the state itself. Otherwise, we could not average over the same phases anymore.