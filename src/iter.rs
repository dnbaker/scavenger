use tch::{Device, IndexOp, Kind, TchError, Tensor};

/// An iterator over a pair of tensors which have the same first dimension
/// size.
/// The typical use case is to iterate over batches. Each batch is a pair
/// containing a (potentially random) slice of each of the two input
/// tensors.
#[derive(Debug)]
pub struct Iter {
    xs: Tensor,
    ys: Option<Tensor>,
    batch_index: i64,
    batch_size: i64,
    total_size: i64,
    device: Device,
    return_smaller_last_batch: bool,
}

impl Iter {
    /// Returns a new iterator.
    ///
    /// This takes as input two tensors which first dimension must match. The
    /// returned iterator can be used to range over mini-batches of data of
    /// specified size.
    /// An error is returned if `xs` and `ys` have different first dimension
    /// sizes.
    ///
    /// # Arguments
    ///
    /// * `xs` - the features to be used by the model.
    /// * `ys` - the targets that the model attempts to predict.
    /// * `batch_size` - the size of batches to be returned.
    pub fn f_new(xs: &Tensor, ys: Option<&Tensor>, batch_size: i64) -> Result<Iter, TchError> {
        let total_size = xs.size()[0];
        if ys.is_some() && ys.unwrap().size()[0] != total_size {
            return Err(TchError::Shape(format!(
                "different dimension for the two inputs {xs:?} {ys:?}"
            )));
        }
        Ok(Iter {
            xs: xs.shallow_clone(),
            ys: ys.map(|x| x.shallow_clone()),
            batch_index: 0,
            batch_size,
            total_size,
            device: Device::Cpu,
            return_smaller_last_batch: false,
        })
    }

    /// Returns a new iterator.
    ///
    /// This takes as input two tensors which first dimension must match. The
    /// returned iterator can be used to range over mini-batches of data of
    /// specified size.
    /// Panics if `xs` and `ys` have different first dimension sizes.
    ///
    /// # Arguments
    ///
    /// * `xs` - the features to be used by the model.
    /// * `ys` - the targets that the model attempts to predict.
    /// * `batch_size` - the size of batches to be returned.
    pub fn new(xs: &Tensor, ys: Option<&Tensor>, batch_size: i64) -> Iter {
        Iter::f_new(xs, ys, batch_size).unwrap()
    }

    /// Shuffles the dataset.
    ///
    /// The iterator would still run over the whole dataset but the order in
    /// which elements are grouped in mini-batches is randomized.
    pub fn shuffle(&mut self) -> &mut Iter {
        let index = Tensor::randperm(self.total_size, (Kind::Int64, self.device));
        self.xs = self.xs.index_select(0, &index);
        self.ys = self.ys.as_ref().map(|x| x.index_select(0, &index));
        self
    }

    /// Transfers the mini-batches to a specified device.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, device: Device) -> &mut Iter {
        self.device = device;
        self
    }

    /// When set, returns the last batch even if smaller than the batch size.
    pub fn return_smaller_last_batch(&mut self) -> &mut Iter {
        self.return_smaller_last_batch = true;
        self
    }
}

impl Iterator for Iter {
    type Item = (Tensor, Option<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let size = std::cmp::min(self.batch_size, self.total_size - start);
        if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
            None
        } else {
            self.batch_index += 1;
            Some((
                self.xs.i(start..start + size).to_device(self.device),
                self.ys
                    .as_ref()
                    .map(|x| x.i(start..start + size).to_device(self.device)),
            ))
        }
    }
}
