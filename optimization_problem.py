class OptimizationProblem:
    """
    This class represents an optimization problem of a
    multidimensional Lipschitz continuous function.
    """

    # The flag 'negative' indicates whether negation should be
    # applied to targetFunction. 
    # This flag is useful because it does not seem straightforward
    # to negate a multidimensional function specified using Python's
    # lambdas constructs. Hence, to negate a multidimensional funciton,
    # we the function inside an object and store a flag to denote the
    # sign. 
    def __init__(self, targetFunction, domain, negative=False):
        self.targetFunction = targetFunction
        self.domain = domain
        self.negative = negative

        assert self.isValidDomain(domain)

    def isValidDomain(self, domain):
        """
        This method returns true if the given domain is valid."""
        for (a,b) in domain:
            if a >= b:
                return False
        return True

    def getArity(self):
        """
        This method returns the arity of targetFunction.
        """
        return len(self.domain)
    
    def evaluate(self, values):
        """
        This method evaluates targetFunction at the coordinate
        specified by the list 'values'. 
        """
        if len(values) != self.getArity():
            raise Exception("Not all arguments have been supplied yet.")
        elif self.negative:
            return - self.targetFunction(*values)
        else:
            return self.targetFunction(*values)

    def getInverse(self):
        """
        This method returns a new OptimizationProblem object
        with negation.
        """
        return OptimizationProblem(self.targetFunction, self.domain, negative=True)
    