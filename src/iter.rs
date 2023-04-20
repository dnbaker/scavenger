use tch::{Device, IndexOp, Kind, TchError, Tensor};

use std::iter::ExactSizeIterator;
use std::ops::Range;

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
    pub fn shuffle_sparse(&mut self, sparse: bool) -> &mut Iter {
        if !sparse {
            self.shuffle();
        }
        self
    }
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

#[derive(Debug)]
pub struct CSRMatrix {
    pub data: Tensor,
    pub indices: Tensor,
    pub indptr: Tensor,
    pub shape: Vec<i64>,
    pub kind: Kind,
}

impl CSRMatrix {
    pub fn size(&self) -> &[i64] {
        &self.shape[..]
    }
    pub fn shallow_clone(&self) -> CSRMatrix {
        Self {
            data: self.data.shallow_clone(),
            indices: self.indices.shallow_clone(),
            indptr: self.indptr.shallow_clone(),
            shape: self.shape.clone(),
            kind: self.kind,
        }
    }
    pub fn extract_range(&self, range: Range<i64>) -> Tensor {
        let numrows = range.end - range.start;
        let indptr: Vec<i64> = Vec::<i64>::from(self.indptr.i(range.clone()));
        /*
        let indptr_start = *indptr.first().unwrap();
        let indptr_stop = *indptr.last().unwrap();
        let mut data_indices: Vec<i64> = vec![0i64; nnz as usize];
        indptr.windows(2).enumerate().for_each(|(idx, slice)| {
            let from = (slice[0] - indptr_start) as usize;
            let to = (slice[1] - indptr_start) as usize;
            data_indices[from..to].fill(idx as i64);
        });
        let sparse_indices: Range<i64> = Range {
            start: indptr_start,
            end: indptr_stop,
        };
        let data_indices = Tensor::of_slice(data_indices.as_slice());
        let col_indices = self.indices.i(sparse_indices.clone()).to_kind(Kind::Int);
        let data = self.data.i(sparse_indices).to_kind(self.kind);
        // Not a true CSR, but I still have to use the api...
        let sparse_mat = tch::Tensor::sparse_csr_tensor_crow_col_value_size(
            &data_indices,
            &col_indices,
            /*values=*/ &data,
            &self.shape[..],
            (self.kind, self.data.device()),
        );
        log::info!("Mat: {:?}. Kind: {:?}", sparse_mat, self.kind);
        sparse_mat.to_dense(self.kind)
        */
        // TODO: make this just one index_put_ call somehow.
        let mut mat = Tensor::zeros(&[numrows, self.shape[1]], (self.kind, self.data.device()));
        for (row_idx, slice) in indptr.windows(2).enumerate() {
            log::debug!("row {} and slice {:?}", row_idx, slice);
            let slice = (slice[0] as i64)..(slice[1] as i64);
            let slice_len = slice.end - slice.start;
            let col_idx = self.indices.i(slice.clone()).to_kind(Kind::Int);
            let data_val = self.data.i(slice);
            let xvals = self.data.new_full(
                &[slice_len],
                row_idx as i64,
                (Kind::Int64, self.data.device()),
            );
            mat = mat.index_put_(&[Some(xvals), Some(col_idx)], &data_val, false);
            /*
            for (col_idx, data_val) in col_idx.zip(data_val) {
                mat.index_put_(&[Some(row_idx as i64), Some(col_idx)], Tensor::from(data_val), false);
                //mat.i((row_idx as i64, col_idx)) = data_val;
            }
            */
        }
        log::debug!(
            "Mat: {mat}, {:?}. Sum: {}",
            mat.size(),
            mat.sum(Kind::Double)
        );
        mat
    }
}

#[derive(Debug)]
pub struct IterCSR {
    xs: CSRMatrix,
    ys: Option<Tensor>,
    batch_index: i64,
    batch_size: i64,
    total_size: i64,
    device: Device,
    return_smaller_last_batch: bool,
}

impl IterCSR {
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
    pub fn f_new(
        xs: &CSRMatrix,
        ys: Option<&Tensor>,
        batch_size: i64,
    ) -> Result<IterCSR, TchError> {
        let total_size = xs.size()[0];
        if ys.is_some() && ys.unwrap().size()[0] != total_size {
            return Err(TchError::Shape(format!(
                "different dimension for the two inputs {xs:?} {ys:?}"
            )));
        }
        Ok(IterCSR {
            xs: xs.shallow_clone(),
            ys: ys.map(|x| x.shallow_clone()),
            batch_index: 0,
            batch_size,
            total_size,
            device: Device::Cpu,
            return_smaller_last_batch: false,
        })
    }
    pub fn new(xs: &CSRMatrix, ys: Option<&Tensor>, batch_size: i64) -> IterCSR {
        IterCSR::f_new(xs, ys, batch_size).unwrap()
    }
    /// Transfers the mini-batches to a specified device.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, device: Device) -> &mut Self {
        self.device = device;
        self
    }
}

impl Iterator for IterCSR {
    type Item = (Tensor, Option<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let size = std::cmp::min(self.batch_size, self.total_size - start);
        if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
            None
        } else {
            self.batch_index += 1;
            Some((
                self.xs
                    .extract_range(start..start + size)
                    .to_device(self.device),
                self.ys
                    .as_ref()
                    .map(|x| x.i(start..start + size).to_device(self.device)),
            ))
        }
    }
}
