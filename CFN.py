#import pytoulbar2 as tb2
try :
    import pytoulbar2.pytb2 as tb2
except :
    import pytoulbar2 as tb2


class CFN:
    def __init__(self, ubinit, resolution=6, vac=True, backtrack=100000, allSolutions = 0, verbose = -1):
        tb2.init()
        tb2.option.decimalPoint = resolution   # decimal precision of costs
        # if no zero, maximum search depth-1 where VAC algorithm is performed (use 1 for preprocessing only)
        tb2.option.vac = vac
        # if True, exploit VAC integrality variable ordering heuristic or just Full-EAC heuristic if VAC diseable
        tb2.option.FullEAC = True
#        tb2.option.VACthreshold = True     # if True, reuse VAC auto-threshold value found in preprocessing during search
        tb2.option.useRASPS = True         # if True, perform RASPS algorithm in preprocessing to find good initial upperbound (use with VAC and FullEAC)
#        tb2.option.hbfs = False            # if True, apply hybrid best-first search algorithm, else apply depth-first search algorithm
        # maximum number of backtracks before restart
        tb2.option.backtrackLimit = backtrack
        # variable ordering heuristic exploiting cost distribution information (0: none, 1: mean cost, 2: median cost)
        tb2.option.weightedTightness = True
        # verbosity level of toulbar2 (-1:no message, 0:search statistics, 1:search tree, 2-7: propagation information)
        tb2.option.verbose = verbose
        # show solutions found (0: none, 1: value indexes, 2: value names, 3: variable and value names if available)
        tb2.option.showSolutions = 0
        # start with a quick INCOP local search
        #tb2.option.incop_cmd = "0 1 3 idwa 100000 cv v 0 200 1 0 0"
        tb2.option.allSolutions = allSolutions  # 000     # find all solutions up to a maximum limit

        self.Variables = {}
        self.VariableIndices = {}
        self.Scopes = []
        self.VariableNames = []
        # initialize VAC algorithm depending on tb2.option.vac
        self.CFN = tb2.Solver(ubinit * 10**resolution)

        self.Contradiction = tb2.Contradiction
        self.SolverOut = tb2.SolverOut
        self.Option = tb2.option
        tb2.check()

    def __del__(self):
        del self.Scopes
        del self.Variables
        del self.VariableIndices
        del self.VariableNames
        del self.CFN

    def NoPreprocessing(self):
        tb2.option.elimDegree = -1
        tb2.option.elimDegree_preprocessing = -1
        tb2.option.preprocessTernaryRPC = 0
        tb2.option.preprocessFunctional = 0
        tb2.option.costfuncSeparate = False
        tb2.option.preprocessNary = 0
        tb2.option.DEE = 0
        tb2.option.MSTDAC = False
        tb2.option.trwsAccuracy = -1

    def AddVariable(self, name, values):
        if name in self.Variables:
            raise RuntimeError(name+" already defined")
        cardinality = len(values)
        self.Variables[name] = values

        if all(isinstance(value, int) for value in values):
            vIdx = self.CFN.wcsp.makeEnumeratedVariable(
                name, min(values), max(values))
            for vn in values:
                self.CFN.wcsp.addValueName(vIdx, 'v' + str(vn))
        elif all(isinstance(value, str) for value in values):
            vIdx = self.CFN.wcsp.makeEnumeratedVariable(name, 0, len(values)-1)
            for vn in values:
                self.CFN.wcsp.addValueName(vIdx, vn)
        else:
            raise RuntimeError("Incorrect domain:"+str(values))
        self.VariableIndices[name] = vIdx
        self.VariableNames.append(name)
        return vIdx

    def AddFunction(self, scope, costs):
        sscope = set(scope)
        if len(scope) != len(sscope):
            raise RuntimeError("Duplicate variable in scope:"+str(scope))
        arity = len(scope)
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices[v]
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v)

        if (len(iscope) == 0):
            self.CFN.wcsp.postNullaryConstraint(costs)
        elif (len(iscope) == 1):
            self.CFN.wcsp.postUnaryConstraint(iscope[0], costs, False)
        elif (len(iscope) == 2):
            self.CFN.wcsp.postBinaryConstraint(
                iscope[0], iscope[1], costs, False)
        elif (len(iscope) == 3):
            self.CFN.wcsp.postTernaryConstraint(
                iscope[0], iscope[1], iscope[2], costs, False)
        else:
            mincost = min(costs)
            maxcost = max(costs)
            self.CFN.wcsp.postNullaryConstraint(mincost)
            if (mincost == maxcost):
                return
            idx = self.CFN.wcsp.postNaryConstraintBegin(
                iscope, 0, len(costs) - costs.count(0), True)
            tuple = [self.CFN.wcsp.toValue(v, 0) for v in iscope]
            for cost in costs:
                if cost > mincost:
                    self.CFN.wcsp.postNaryConstraintTuple(
                        idx, tuple, (cost-mincost) * 10 ** tb2.option.decimalPoint)
                for r in range(len(iscope)):
                    i = len(iscope)-1-r
                    v = iscope[i]
                    if tuple[i] < self.CFN.wcsp.toValue(v, self.CFN.wcsp.getDomainInitSize(v) - 1):
                        tuple[i] += 1
                        for j in range(i+1, len(iscope)):
                            tuple[j] = self.CFN.wcsp.toValue(iscope[j], 0)
                        break
            self.CFN.wcsp.postNaryConstraintEnd(idx)
        self.Scopes.append(sscope)
        return

    def Read(self, problem):
        self.CFN.read(problem)

    def Parse(self, certificate):
        # WARNING! False: do not reuse certificate in future searches used by structure learning evaluation procedure!
        self.CFN.parse_solution(certificate, False)

    def Domain(self, varIndex):
        return self.CFN.wcsp.getEnumDomain(varIndex)

    def GetUB(self):
        return self.CFN.wcsp.getDPrimalBound()

    # decreasing upper bound only
    def UpdateUB(self, cost):
        icost = self.CFN.wcsp.DoubletoCost(cost)
        self.CFN.wcsp.updateUb(icost)
        self.CFN.wcsp.enforceUb()   # this might generate a Contradiction exception

    # important! used in incremental solving for changing initial upper bound up and down before adding any problem modifications
    def SetUB(self, cost):
        icost = self.CFN.wcsp.DoubletoCost(cost)
        self.CFN.wcsp.setUb(icost)  # must be done after problem loading
        # important to notify previous best found solution is no more valid
        self.CFN.wcsp.initSolutionCost()
        self.CFN.wcsp.enforceUb()   # this might generate a Contradiction exception

    def Depth(self):
        return tb2.store.getDepth()

    # make a copy of the current problem and move to store.depth+1
    def Store(self):
        tb2.store.store()

    # restore previous copy made at a given depth
    def Restore(self, depth):
        tb2.store.restore(depth)

    def GetNbNodes(self):
        return self.CFN.getNbNodes()

    def GetNbBacktracks(self):
        return self.CFN.getNbBacktracks()

    # non-incremental solving method
    def Solve(self):
        self.CFN.wcsp.sortConstraints()

        solved = self.CFN.solve()
        if (solved):
            return self.CFN.solution(), self.CFN.wcsp.getDPrimalBound(), len(self.CFN.solutions())
        else:
            return None

    # incremental solving: perform initial preprocessing before all future searches, return improved ub
    def SolveFirst(self):
        self.CFN.wcsp.sortConstraints()
        ub = self.CFN.wcsp.getUb()
        self.CFN.beginSolve(ub)
        try:
            ub = self.CFN.preprocessing(ub)
        except tb2.Contradiction:
            self.CFN.wcsp.whenContradiction()
            print('Problem has no solution!')
            return None
        return self.CFN.wcsp.Cost2ADCost(ub)

    # incremental solving: find the next optimum value after a problem modification (see also SetUB)
    def SolveNext(self):
        initub = self.CFN.wcsp.getUb()
        initdepth = tb2.store.getDepth()
        self.CFN.beginSolve(initub)
        # reinitialize this parameter which can be modified during hybridSolve()
        tb2.option.hbfs = 1
        try:
            try:
                tb2.store.store()
                self.CFN.wcsp.propagate()
                lb, ub = self.CFN.hybridSolve()
            except tb2.Contradiction:
                self.CFN.wcsp.whenContradiction()
        except tb2.SolverOut:
            tb2.option.limit = False
        tb2.store.restore(initdepth)
        if self.CFN.wcsp.getSolutionCost() < initub:
            #  warning! None: does not return number of found solutions because it is two slow to retrieve all solutions in python
            return self.CFN.solution(), self.CFN.wcsp.getDPrimalBound(), None
        else:
            return None

    def Dump(self, problem):
        if '.wcsp' in problem:
            self.CFN.dump_wcsp(problem, True, 1)
        elif '.cfn' in problem:
            self.CFN.dump_wcsp(problem, True, 2)
        else:
            print('Error unknown format!')
            
    def GetSolutions(self):
        """GetSolutions returns all the solutions found so far with their associated costs.

        Returns:
            List of pairs (decimal cost, solution) where a solution is a list of domain values.

        """
        return self.CFN.solutions()
