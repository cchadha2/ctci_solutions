class ThreeStacks:

    def __init__(self, array_len):
        self.arr = [None] * array_len
        divisor = array_len // 3
        self.ranges = ((0, divisor), (divisor, 2*divisor), (2*divisor, array_len))
        self.curr_idx = [elem[0] for elem in self.ranges]

    def peek(self, stack):
        return self.arr[self.curr_idx[stack]]

    def is_empty(self, stack):
        return ((self.arr[self.curr_idx[stack]] is None)
                 and (self.curr_idx[stack] == self.ranges[stack][0]))

    def push(self, stack, value):
        if ((self.curr_idx[stack] == self.ranges[stack][1] - 1) and
            (self.arr[self.curr_idx[stack]] is not None)):
            raise ValueError("Stack Overflow")

        if not self.is_empty(stack) or self.curr_idx[stack] != self.ranges[stack][1] - 1:
            self.curr_idx[stack] += 1

        self.arr[self.curr_idx[stack]] = value

    def pop(self, stack):
        if self.is_empty(stack):
            raise ValueError("Can't pop from empty stack")

        return_val = self.arr[self.curr_idx[stack]]
        self.arr[self.curr_idx[stack]] = None
        if not self.is_empty(stack):
            self.curr_idx[stack] -= 1

        return return_val


if __name__ == "__main__":
    length = 15
    stacks = ThreeStacks(length)

    print(stacks.push(0, 15))

    print(stacks.peek(0))
    print(stacks.is_empty(1))

    print(stacks.pop(0))
    print(stacks.is_empty(0))
