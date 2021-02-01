from numpy import random, linalg

# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def random_in_ball(num_points, dimension, radius=1, norm="l2"):

    if norm == "l2":
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = random.normal(size=(dimension,num_points))
        random_directions /= linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = random.random(num_points) ** (1/dimension)
        # Return the list of random (direction & length) points.
        res = radius * (random_directions * random_radii).T

    elif norm == "infinity":
        res = random.uniform(low=-radius,high=radius,size=(num_points,dimension))

    return res


def random_in_sphere(num_points, dimension, radius=1, norm="l2"):
    from numpy import random, linalg

    if norm == "l2":
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = random.normal(size=(dimension,num_points))
        random_directions /= linalg.norm(random_directions, axis=0)
        res = radius * random_directions.T

    #elif norm == "infinity":
        #res = random.uniform(low=-radius,high=radius,size=(num_points,dimension))
    return res