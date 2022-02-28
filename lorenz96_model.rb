require "narray"

class Lorenz96
  def initialize(k, f, dt, nt)
    @k = k
    @f = f
    @dt = dt
    @nt = nt
    @idxn2 = NArray.sint(k).indgen(-2)
    @idxn1 = NArray.sint(k).indgen(-1)
    tmp = NArray.sint(k).indgen(1)
    tmp[-1] = 0
    @idxp1 = tmp
    tmp = NArray.sint(k).indgen(2)
    tmp[-2] = 0
    tmp[-1] = 1
    @idxp2 = tmp
  end

  def derivative(x)
    dx = x[@idxn1] * ( x[@idxp1] - x[@idxn2] ) - x + @f
    return dx
  end

  def forward(x)
    d1 = derivative(x)
    d2 = derivative(x + d1 * ( @dt * 0.5 ) )
    d3 = derivative(x + d2 * ( @dt * 0.5 ) )
    d4 = derivative(x + d3 * @dt)
    x = x + ( d1 + ( d2 + d3 ) * 2.0 + d4 ) * ( @dt / 6.0 )
    return x
  end

  def step(x)
    @nt.times do
      x = forward(x)
    end
    x
  end

end

