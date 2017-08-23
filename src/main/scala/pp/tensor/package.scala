package pp

package object tensor {

  sealed trait Nat {
    type P <: Nat
    type Add[A <: Nat] <: Nat // 1.add(5)
    type Sub[A <: Nat] <: Nat // 5.sub(1)
  }

  object Nat {
    type _0 = Zero
    type _1 = Succ[Zero]
    type _2 = Succ[_1]
    type _3 = Succ[_2]
    type _4 = Succ[_3]
    type _5 = Succ[_4]
    type _6 = Succ[_5]
  }

  case class Zero() extends Nat {
    type P = Zero
    type Add[A <: Nat] = A
    type Sub[A <: Nat] = A
  }

  case class Succ[N <: Nat]() extends Nat {
    type P = N
    type Add[A <: Nat] = Succ[N#Add[A]]
    type Sub[A <: Nat] = Succ[N#Sub[A#P]]
  }

  type Plus[A <: Nat, B <: Nat] = A#Add[B]
  type Min[A <: Nat, B <: Nat] = A#Sub[B]

  trait ToInt[N <: Nat] {
    def apply(): Int
  }

  object ToInt {
    implicit val zeroInt: ToInt[Zero] = new ToInt[Zero] {
      override def apply() = 0
    }

    implicit def succToInt[N <: Nat](implicit prev: ToInt[N]): ToInt[Succ[N]] = new ToInt[Succ[N]] {
      override def apply() = prev.apply() + 1
    }
  }

}
