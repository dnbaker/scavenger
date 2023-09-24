1. Simplify the NBVAE aspect.
   Return separate mean, var, mu, etc.
   Take the class loss out of the NBVAE and handle it separately.
   Scaled reconstruction loss is very much a win.
2. Learn to use PyOT to map between data types, then use contrastive learning.
3. Think of contrastive learning generalizations that take into account relative properties.
