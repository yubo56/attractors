# Yubo dynamics work

Generic folder for work concerning separatrix hopping, for disks and for weak
tides, and maybe some actual attractors. References are in my Dropbox folder

## 1weaktide notes

Taking some notes here
On sc = 0.06, seems like some states get stuck, e.g. for IC (-0.506, 6.189),
    stuck on t, svec, s = (
    785.7484653150983,
    [ 0.72318343 -0.33824331 -0.60216044]
    2.9731638916732805)

On `5outcomes0_20.png`, there seem to be two tracks.
  - In idx1:
    - bottom track fiducial IC is -0.9622609, 3.1416017
        - Is case where s spins down to < s_c, so eta > 1, then spin
          re-increases back so ends up at CS2
    - top track is the correct one, just points that start within separatrix,
      e.g. -0.031588, 4.008468
  - In idx2:
    - Do higher points really show sep crossing? e.g. 0.7515566, 4.845555
    - How does the lower island synchronize? e.g. -0.87422, 2.92419
    - mu in [-0.7, 0] behave as expected, probabilistically go to CS1
  - In idx3, again, what is this lower island doing? The rest make sense
    - lower island e.g. (-0.777320757, 1.95619)

On `5outcomes0_60.png`, idx1 make sense as the points that do not undergo sep
cross, either to CS2 (spin is so low it never sees a sep) or go straight up to
CS2. idx3 on the other hand:
  - 0.9172209, 2.01255 is an example; think it's just a very late sep encounter?
