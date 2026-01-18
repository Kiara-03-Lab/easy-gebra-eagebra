"""
Algebra MVP Backend
FastAPI + SymPy for polynomial operations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

app = FastAPI(title="Algebra Calculator API")

# Allow all origins for MVP (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parsing transformations for user-friendly input
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


class ComputeRequest(BaseModel):
    poly1: str
    poly2: Optional[str] = None
    operation: str  # factor, gcd, divide, evaluate, expand, simplify
    eval_point: Optional[dict] = None  # e.g., {"x": 2, "y": 3}
    field: str = "QQ"  # QQ (rationals), ZZ (integers), or "Zp:7" for Z/7Z


class ComputeResponse(BaseModel):
    success: bool
    latex: Optional[str] = None
    plaintext: Optional[str] = None
    error: Optional[str] = None


def parse_polynomial(expr_str: str, field: str = "QQ"):
    """Parse user input into SymPy expression."""
    try:
        # Clean input
        expr_str = expr_str.strip()
        # Convert ^ to ** for Python exponentiation
        expr_str = expr_str.replace('^', '**')
        expr = parse_expr(expr_str, transformations=TRANSFORMS)
        return expr
    except Exception as e:
        raise ValueError(f"Cannot parse '{expr_str}': {str(e)}")


def get_domain(field: str):
    """Get SymPy domain from field string."""
    if field == "QQ":
        return sp.QQ
    elif field == "ZZ":
        return sp.ZZ
    elif field.startswith("Zp:"):
        p = int(field.split(":")[1])
        return sp.GF(p)
    else:
        return sp.QQ


@app.get("/")
def health_check():
    return {"status": "ok", "service": "algebra-calculator"}


@app.post("/compute", response_model=ComputeResponse)
def compute(req: ComputeRequest):
    try:
        poly1 = parse_polynomial(req.poly1, req.field)
        domain = get_domain(req.field)
        
        result = None
        
        if req.operation == "factor":
            if req.field.startswith("Zp:"):
                p = int(req.field.split(":")[1])
                result = sp.factor(poly1, modulus=p)
            else:
                result = sp.factor(poly1)
                
        elif req.operation == "expand":
            result = sp.expand(poly1)
            
        elif req.operation == "simplify":
            result = sp.simplify(poly1)
            
        elif req.operation == "gcd":
            if not req.poly2:
                raise ValueError("GCD requires two polynomials")
            poly2 = parse_polynomial(req.poly2, req.field)
            result = sp.gcd(poly1, poly2)
            
        elif req.operation == "divide":
            if not req.poly2:
                raise ValueError("Division requires two polynomials")
            poly2 = parse_polynomial(req.poly2, req.field)
            # Get quotient and remainder
            x = list(poly1.free_symbols)[0] if poly1.free_symbols else sp.Symbol('x')
            q, r = sp.div(poly1, poly2, x)
            # Format as "quotient, remainder r"
            result = f"{sp.latex(q)} \\text{{ remainder }} {sp.latex(r)}"
            return ComputeResponse(
                success=True,
                latex=result,
                plaintext=f"({q}, {r})"
            )
            
        elif req.operation == "evaluate":
            if not req.eval_point:
                raise ValueError("Evaluation requires point values")
            # Convert string keys to symbols
            subs = {sp.Symbol(k): v for k, v in req.eval_point.items()}
            result = poly1.subs(subs)
            
        elif req.operation == "derivative":
            # Find variable (default to x)
            symbols = list(poly1.free_symbols)
            var = symbols[0] if symbols else sp.Symbol('x')
            result = sp.diff(poly1, var)
            
        elif req.operation == "roots":
            result = sp.solve(poly1)
            latex_roots = ", ".join([sp.latex(r) for r in result])
            return ComputeResponse(
                success=True,
                latex=f"\\{{ {latex_roots} \\}}",
                plaintext=str(result)
            )
            
        else:
            raise ValueError(f"Unknown operation: {req.operation}")
        
        return ComputeResponse(
            success=True,
            latex=sp.latex(result),
            plaintext=str(result)
        )
        
    except ValueError as e:
        return ComputeResponse(success=False, error=str(e))
    except Exception as e:
        return ComputeResponse(success=False, error=f"Computation error: {str(e)}")


# Additional endpoint for quick operations
@app.get("/factor/{polynomial}")
def quick_factor(polynomial: str):
    """Quick factor endpoint for simple GET requests."""
    try:
        expr = parse_polynomial(polynomial)
        result = sp.factor(expr)
        return {
            "input": polynomial,
            "latex": sp.latex(result),
            "plaintext": str(result)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
