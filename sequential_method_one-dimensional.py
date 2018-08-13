# This script implements the sequential global optimization method
# for one-dimensional Lipschitz continuous functions. 

class OneDimentionalSequentialMethod:

    # targetFunction is the function whose global extrema are to be computed.
    # a and b are the left and right bounds of the domain of targetFunction
    # over which the global extrema are sought. 
    # eta is a reliability parameter that is used when K is dynamically estimated. 
    # K is a Lipschitz constant of targetFunction. 
    def __init__(self, targetFunction, a, b, K=None, eta=1.5, minimumIteration=1000, maximumIteration=10000):
        
        # The reliabiity parameter needs to be larger than 1.
        # Also, the left and right bounds of the domain must be distinct. 
        assert eta > 1 and a != b
        
        self.targetFunction = targetFunction
        self.a = a
        self.b = b
        self.K = K
        self.eta = eta
        self.minimumIteration = minimumIteration
        self.maximumIteration = maximumIteration

    def computeMinimum(self, errorBound):
        """This method computes the global minimum of self.targetFunction."""
        if self.K is None:
            return self.computeMinimumOfFunctionDynamic(self.targetFunction, self.a, self.b, self.eta, errorBound, self.maximumIteration)
        else:
            return self.computeMinimumOfFunctionStatic(self.targetFunction, self.a, self.b, self.K, errorBound, self.maximumIteration)
        
    def computeMaximum(self, errorBound):
        """This method computes the global maximum of self.taregtFunction."""
        if self.K is None:
            return - self.computeMinimumOfFunctionDynamic(lambda x: - self.targetFunction(x), self.a, self.b, self.eta, errorBound, self.maximumIteration)    
        else:
            return - self.computeMinimumOfFunctionStatic(lambda x: - self.targetFunction(x), self.a, self.b, self.K, errorBound, self.maximumIteration)

    def computeMinimumOfFunctionStatic(self, f, a, b, K, errorBound, maximumIteration):
        """
        This method computes the global minimum of f over [a,b].
        A Lipschitz constant is given by a caller.
        """

        # The left and right bounds of the domain must be distinct.
        assert a != b

        upperBound = min(f(a), f(b))
        lowerBound = self.computeNewZ(f, a, b, K)
        ys = [a, b]
        zs = [lowerBound]
        
        # Counter for the while loop
        i = 0

        # Index of zs at which the smallest z occurs. 
        index = 0 
        while i != maximumIteration and upperBound - lowerBound >= errorBound:
            newY = self.computeNewY(f, ys[index], ys[index + 1], K)

            # If the differences between newY, ys[index], and ys[idnex + 1] are so small that
            # they become 0, then the loop needs to be broken in order to avoid division by 0.
            if newY == ys[index] or newY == ys[index + 1]:
                break

            newZLeft = self.computeNewZ(f, ys[index], newY, K)
            newZRight = self.computeNewZ(f, newY, ys[index + 1], K)

            ys.insert(index + 1, newY)
            zs[index] = newZLeft
            zs.insert(index + 1, newZRight)

            upperBound = min(upperBound, f(newY))
            index = self.indexOfMinimumZ(zs)
            lowerBound = zs[index]
            i += 1
        
        return (upperBound + lowerBound) / 2

    def computeMinimumOfFunctionDynamic(self, f, a, b, eta, errorBound, maximumIteration):
        """
        This method computes the global minimum of f over [a,b].
        A Lipschitz constant is dynamically estimated.
        """

        # The left and right bounds of the domain must be distinct.
        # Also, eta should be larger than 1.
        assert a != b and eta > 1

        K = max(1, eta * abs((f(b) - f(a)) / (b - a)))

        upperBound = min(f(a), f(b))
        lowerBound = self.computeNewZ(f, a, b, K)
        ys = [a, b]
        zs = [lowerBound]
        
        # Counter for the while loop
        i = 0

        # Index of zs at which the smallest z occurs. 
        index = 0
        while i < self.minimumIteration or (i != maximumIteration and upperBound - lowerBound >= errorBound):
            newY = self.computeNewY(f, ys[index], ys[index + 1], K)

            # If the differences between newY, ys[index], and ys[idnex + 1] are so small that
            # they become 0, then the loop needs to be broken in order to avoid division by 0.
            if newY == ys[index] or newY == ys[index + 1]:
                break

            ys.insert(index + 1, newY)

            # Dynamically update K
            gradientLeft = eta * abs((f(newY) - f(ys[index])) / (newY - ys[index]))
            gradientRight = eta * abs((f(ys[index + 2]) - f(newY)) / (ys[index + 2] - newY))
            if K < gradientLeft or K < gradientRight:
                K = max(gradientLeft, gradientRight)
                zs = self.recomputeZs(f, ys, K)
            else:
                newZLeft = self.computeNewZ(f, ys[index], newY, K)
                newZRight = self.computeNewZ(f, newY, ys[index + 2], K)

                zs[index] = newZLeft
                zs.insert(index + 1, newZRight)

            upperBound = min(upperBound, f(newY))
            index = self.indexOfMinimumZ(zs)
            lowerBound = zs[index]
            i += 1
        
        return (upperBound + lowerBound) / 2

    def recomputeZs(self, f, ys, K):
        """This method creates a new list of zs using ys and a newly updated K."""
        return [self.computeNewZ(f, ys[i], ys[i + 1], K) for i in range(len(ys) - 1)]

    def computeNewY(self, f, u, v, K):
        """This method computes E_X (u, v)."""
        return (u + v) / 2 - (f(u) - f(v)) / (2 * K)

    def computeNewZ(self, f, u, v, K):
        """This method computes E_Y (u, v)."""
        return (f(u) + f(v)) / 2 - K * (v - u) / 2

    def indexOfMinimumZ(self, zs):
        """
        This method returns the index of list zs at which the minimum
        element occurs.
        """
        minimum = zs[0]
        index = 0
        i = 0
        while i != len(zs):
            if zs[i] < minimum:
                index = i
                minimum = zs[i]
            i += 1
        return index

# Testing
if __name__ == "__main__":
    optimizer = OneDimentionalSequentialMethod(lambda x: x ** 3 - 3 * (x ** 2) + 5, -5, 5)
    minimum = optimizer.computeMinimum(0.001)
    maximum = optimizer.computeMaximum(0.001)
    print("The minimum is", minimum)
    print("The maxium is", maximum)
