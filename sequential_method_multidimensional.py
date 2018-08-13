from optimization_problem import OptimizationProblem

# This script implements the sequential global optimization method
# for multidimensional Lipschitz continuous functions. 

class MultidimensionalSequentialMethod:

    def __init__(self, optimizationProblem, eta=1.5, K=None, minimumIteration=3, maximumIteration=5):

        assert eta > 1
        
        self.optimizationProblem = optimizationProblem
        self.eta = eta
        self.K = K
        self.minimumIteration = minimumIteration
        self.maximumIteration = maximumIteration
    
    def computeMinimum(self, errorBound):
        """
        This method computes the global minimum of the function
        stored inside self.optimizationProblem.
        """
        if self.K is None:
            return self.computeMinimumOfFunctionDynamic(self.optimizationProblem, [], self.eta, errorBound, self.maximumIteration)
        else:
            return self.computeMinimumOfFunctionStatic(self.optimizationProblem, [], self.K, errorBound, self.maximumIteration)

    def computeMaximum(self, errorBound):
        """
        This method computes the global maximum of the function
        stored inside self.optimizationProblem.
        """
        inverse = self.optimizationProblem.getInverse()
        if self.K is None:
            return - self.computeMinimumOfFunctionDynamic(inverse, [], self.eta, errorBound, self.maximumIteration)
        else:
            return - self.computeMinimumOfFunctionStatic(inverse, [], self.K, errorBound, self.maximumIteration)

    def computeMinimumOfFunctionStatic(self, optimizationProblem, values, K, errorBound, maximumIteration):
        """
        This method computes the global minimum of the function encapsulated
        inside optimizationProblem.
        'values' is the list of parameters that have been supplied so far
        to the target function. 
        A Lipschitz constant is given by a caller.
        """

        # The number of arguments that have been supplied so far
        # to the target function
        numberOfPastArguments = len(values)
        if numberOfPastArguments == optimizationProblem.getArity():
            return optimizationProblem.evaluate(values)
        
        (a, b) = optimizationProblem.domain[numberOfPastArguments]
        f_a = self.evaluateFunctionStatic(optimizationProblem, values, a, K, errorBound, maximumIteration)
        f_b = self.evaluateFunctionStatic(optimizationProblem, values, b, K, errorBound, maximumIteration)
        ys = [a, b]
        fs = [f_a, f_b] # Stores the values of the target function at ys. 
        upperBound = min(f_a, f_b)
        lowerBound = self.computeNewZ(fs, ys, 0, K)
        zs = [lowerBound]
        
        # Counter for the while loop
        i = 0

        # Index of zs at which the smallest z occurs. 
        index = 0 
        while i != maximumIteration and upperBound - lowerBound >= errorBound:
            newY = self.computeNewY(fs, ys, index, K)

            # If the differences between newY, ys[index], and ys[idnex + 1] are so small that
            # they become 0, then the loop needs to be broken in order to avoid division by 0.
            if newY == ys[index] or newY == ys[index + 1]:
                break

            ys.insert(index + 1, newY)
            f_newY = self.evaluateFunctionStatic(optimizationProblem, values, newY, K, errorBound, maximumIteration)
            fs.insert(index + 1, f_newY)

            newZLeft = self.computeNewZ(fs, ys, index, K)
            newZRight = self.computeNewZ(fs, ys, index + 1, K)

            zs[index] = newZLeft
            zs.insert(index + 1, newZRight)

            upperBound = min(upperBound, f_newY)
            index = self.indexOfMinimumZ(zs)
            lowerBound = zs[index]
            i += 1
        
        # Note that the upper bound, instead of the average of the upper bound
        # and the lower bound, is returned. This is because the upper bound
        # is empirically closer to the correct answer than the average. 
        return upperBound

    def computeMinimumOfFunctionDynamic(self, optimizationProblem, values, eta, errorBound, maximumIteration):
        """
        This method computes the global minimum of the function encapsulated
        inside optimizationProblem.
        'values' is the list of parameters that have been supplied so far
        to the target function. 
        A Lipschitz constant is dynamically estimated.
        """
        
        # The number of arguments that have been supplied so far
        # to the target function
        numberOfPastArguments = len(values)
        if numberOfPastArguments == optimizationProblem.getArity():
            return optimizationProblem.evaluate(values)
        
        (a, b) = optimizationProblem.domain[numberOfPastArguments]
        f_a = self.evaluateFunctionDynamic(optimizationProblem, values, a, eta, errorBound, maximumIteration)
        f_b = self.evaluateFunctionDynamic(optimizationProblem, values, b, eta, errorBound, maximumIteration)
        ys = [a, b]
        fs = [f_a, f_b] # Stores the values of the target function at ys. 

        K = max(1, eta * abs((f_b - f_a) / (b - a)))
        upperBound = min(f_a, f_b)
        lowerBound = self.computeNewZ(fs, ys, 0, K)
        zs = [lowerBound]
        
        # Counter for the while loop
        i = 0

        # Index of zs at which the smallest z occurs. 
        index = 0 
        while i < self.minimumIteration or (i != maximumIteration and upperBound - lowerBound >= errorBound):
            newY = self.computeNewY(fs, ys, index, K)

            # If the differences between newY, ys[index], and ys[idnex + 1] are so small that
            # they become 0, then the loop needs to be broken in order to avoid division by 0.
            if newY == ys[index] or newY == ys[index + 1]:
                break

            ys.insert(index + 1, newY)
            f_newY = self.evaluateFunctionDynamic(optimizationProblem, values, newY, K, errorBound, maximumIteration)
            fs.insert(index + 1, f_newY)

            # Dynamically update K
            gradientLeft = eta * abs((f_newY - fs[index]) / (newY - ys[index]))
            gradientRight = eta * abs((fs[index + 2] - f_newY) / (ys[index + 2] - newY))
            if K < gradientLeft or K < gradientRight:
                K = max(gradientLeft, gradientRight)
                zs = self.recomputeZs(fs, ys, K)
            else:
                newZLeft = self.computeNewZ(fs, ys, index, K)
                newZRight = self.computeNewZ(fs, ys, index + 1, K)

                zs[index] = newZLeft
                zs.insert(index + 1, newZRight)

            upperBound = min(upperBound, f_newY)
            index = self.indexOfMinimumZ(zs)
            lowerBound = zs[index]
            i += 1
        
        # Note that the upper bound, instead of the average of the upper bound
        # and the lower bound, is returned. This is because the upper bound
        # is empirically closer to the correct answer than the average. 
        return upperBound

    def recomputeZs(self, fs, ys, K):
        """This method creates a new list of zs using ys and a newly updated K."""
        return [self.computeNewZ(fs, ys, i, K) for i in range(len(ys) - 1)]

    def evaluateFunctionStatic(self, optimizationProblem, values, x, K, errorBound, maximumIteration):
        newValues = values + [x]
        return self.computeMinimumOfFunctionStatic(optimizationProblem, newValues, K, errorBound, maximumIteration)

    def evaluateFunctionDynamic(self, optimizationProblem, values, x, eta, errorBound, maximumIteration):
        newValues = values + [x]
        return self.computeMinimumOfFunctionDynamic(optimizationProblem, newValues, eta, errorBound, maximumIteration)

    def computeNewY(self, fs, ys, i, K):
        """This method computes E_X (ys[i], ys[i + 1])."""
        u = ys[i]
        v = ys[i + 1]
        f_u = fs[i]
        f_v = fs[i + 1]
        return (u + v) / 2 - (f_u - f_v) / (2 * K)

    def computeNewZ(self, fs, ys, i, K):
        """This method computes E_Y (ys[i], ys[i + 1])."""
        u = ys[i]
        v = ys[i + 1]
        f_u = fs[i]
        f_v = fs[i + 1]
        return (f_u + f_v) / 2 - K * (v - u) / 2

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
    targetFunction = lambda x,y,z: x ** 2 - 3 * y + 3 * z**2 - 10
    problem = OptimizationProblem(targetFunction, [(0,1), (0,1), (0,1)])
    optimizer = MultidimensionalSequentialMethod(problem)
    minimum = optimizer.computeMinimum(0.001)
    maximum = optimizer.computeMaximum(0.001)
    print("The minimum is", minimum)
    print("The maxium is", maximum)