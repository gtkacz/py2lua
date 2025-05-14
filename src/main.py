import argparse
import ast
import sys
from typing import Any


class LuaTranspiler:
	def __init__(self):
		self.indentation = 0
		self.current_class = None
		self.imported_modules = set()

	def indent(self):
		self.indentation += 1

	def dedent(self):
		self.indentation -= 1

	def get_indent(self):
		return "  " * self.indentation

	def transpile(self, python_code: str) -> str:
		"""Transpile Python code to Lua code."""
		try:
			tree = ast.parse(python_code)
			return self.visit(tree)
		except SyntaxError as e:
			return f"-- Syntax error: {e}"

	def visit(self, node: Any) -> str:
		"""Visit a node and dispatch to the appropriate method."""
		method_name = f"visit_{node.__class__.__name__}"
		method = getattr(self, method_name, self.generic_visit)
		return method(node)

	def generic_visit(self, node: Any) -> str:
		"""Called if no explicit visitor function exists for a node."""
		return f"-- Unsupported Python construct: {type(node).__name__}"

	def visit_Module(self, node: ast.Module) -> str:
		"""Visit a Module node - the root of the AST."""
		result = []
		for child in node.body:
			result.append(self.visit(child))
		return "\n".join(result)

	def visit_Expr(self, node: ast.Expr) -> str:
		"""Visit an expression statement."""
		return f"{self.get_indent()}{self.visit(node.value)}"

	def visit_Constant(self, node: ast.Constant) -> str:
		"""Visit a constant value."""
		if node.value is None:
			return "nil"
		if isinstance(node.value, bool):
			return str(node.value).lower()
		if isinstance(node.value, (int, float)):
			return str(node.value)
		if isinstance(node.value, str):
			return repr(node.value)
		return f"-- Unsupported constant: {node.value}"

	def visit_Name(self, node: ast.Name) -> str:
		"""Visit a variable name."""
		if node.id == "None":
			return "nil"
		if node.id == "True":
			return "true"
		if node.id == "False":
			return "false"
		if node.id == "self" and self.current_class:
			return "self"
		return node.id

	def visit_Assign(self, node: ast.Assign) -> str:
		"""Visit an assignment statement."""
		value = self.visit(node.value)

		if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
			target = self.visit(node.targets[0])
			return f"{self.get_indent()}{target} = {value}"
		targets = []
		for target in node.targets:
			targets.append(self.visit(target))
		return f"{self.get_indent()}{', '.join(targets)} = {value}"

	def visit_BinOp(self, node: ast.BinOp) -> str:
		"""Visit a binary operation."""
		left = self.visit(node.left)
		right = self.visit(node.right)

		op_map = {
			ast.Add: "+",
			ast.Sub: "-",
			ast.Mult: "*",
			ast.Div: "/",
			ast.FloorDiv: "//",
			ast.Mod: "%",
			ast.Pow: "^",
			ast.LShift: "<<",
			ast.RShift: ">>",
			ast.BitOr: "|",
			ast.BitXor: "~",
			ast.BitAnd: "&",
		}

		op = op_map.get(type(node.op), "-- unknown op")

		# Special case for floor division which is '//' in Python but 'math.floor(a/b)' in Lua
		if isinstance(node.op, ast.FloorDiv):
			return f"math.floor({left} / {right})"

		return f"{left} {op} {right}"

	def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
		"""Visit a unary operation."""
		operand = self.visit(node.operand)

		op_map = {
			ast.USub: "-",
			ast.UAdd: "+",
			ast.Not: "not ",
			ast.Invert: "~",
		}

		op = op_map.get(type(node.op), "-- unknown op")

		return f"{op}{operand}"

	def visit_Compare(self, node: ast.Compare) -> str:
		"""Visit a comparison operation."""
		left = self.visit(node.left)
		comparisons = []

		op_map = {
			ast.Eq: "==",
			ast.NotEq: "~=",
			ast.Lt: "<",
			ast.LtE: "<=",
			ast.Gt: ">",
			ast.GtE: ">=",
			ast.Is: "==",
			ast.IsNot: "~=",
		}

		for i, op in enumerate(node.ops):
			right = self.visit(node.comparators[i])

			if isinstance(op, (ast.In, ast.NotIn)):
				func = "not py_in" if isinstance(op, ast.NotIn) else "py_in"
				comparisons.append(f"{func}({left}, {right})")
			else:
				op_str = op_map.get(type(op), "-- unknown op")
				comparisons.append(f"{left} {op_str} {right}")

			if i < len(node.ops) - 1:
				left = self.visit(node.comparators[i])

		if len(comparisons) == 1:
			return comparisons[0]
		return " and ".join(comparisons)

	def visit_BoolOp(self, node: ast.BoolOp) -> str:
		"""Visit a boolean operation."""
		op_map = {
			ast.And: "and",
			ast.Or: "or",
		}

		op = op_map.get(type(node.op), "-- unknown op")

		values = [self.visit(value) for value in node.values]
		return f" {op} ".join(values)

	def visit_If(self, node: ast.If) -> str:
		"""Visit an if statement."""
		result = []

		condition = self.visit(node.test)
		result.append(f"{self.get_indent()}if {condition} then")

		self.indent()
		for stmt in node.body:
			result.append(self.visit(stmt))
		self.dedent()

		if node.orelse:
			if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
				# This is an elif
				elseif_node = node.orelse[0]
				elseif_condition = self.visit(elseif_node.test)
				result.append(f"{self.get_indent()}elseif {elseif_condition} then")

				self.indent()
				for stmt in elseif_node.body:
					result.append(self.visit(stmt))
				self.dedent()

				if elseif_node.orelse:
					result.append(f"{self.get_indent()}else")
					self.indent()
					for stmt in elseif_node.orelse:
						result.append(self.visit(stmt))
					self.dedent()
			else:
				result.append(f"{self.get_indent()}else")
				self.indent()
				for stmt in node.orelse:
					result.append(self.visit(stmt))
				self.dedent()

		result.append(f"{self.get_indent()}end")

		return "\n".join(result)

	def visit_For(self, node: ast.For) -> str:
		"""Visit a for loop."""
		result = []

		if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
			# Handle range-based for loops
			args = node.iter.args
			target = self.visit(node.target)

			if len(args) == 1:
				# range(stop)
				start = "0"
				stop = self.visit(args[0])
				step = "1"
			elif len(args) == 2:
				# range(start, stop)
				start = self.visit(args[0])
				stop = self.visit(args[1])
				step = "1"
			elif len(args) == 3:
				# range(start, stop, step)
				start = self.visit(args[0])
				stop = self.visit(args[1])
				step = self.visit(args[2])

			# Lua is 1-indexed, Python is 0-indexed for ranges
			# Also, Lua's for loops are inclusive of the end value
			result.append(f"{self.get_indent()}for {target} = {start}, {stop} - 1, {step} do")
		else:
			# Handle generic for loops (using pairs or ipairs)
			target = self.visit(node.target)
			iter_obj = self.visit(node.iter)

			if (
				isinstance(node.iter, ast.Call)
				and isinstance(node.iter.func, ast.Name)
				and node.iter.func.id == "enumerate"
			):
				# Handle enumerate
				iter_obj = self.visit(node.iter.args[0])
				result.append(f"{self.get_indent()}for i, {target} in ipairs({iter_obj}) do")
			else:
				result.append(f"{self.get_indent()}for _, {target} in ipairs({iter_obj}) do")

		self.indent()
		for stmt in node.body:
			result.append(self.visit(stmt))
		self.dedent()

		result.append(f"{self.get_indent()}end")

		return "\n".join(result)

	def visit_While(self, node: ast.While) -> str:
		"""Visit a while loop."""
		result = []

		condition = self.visit(node.test)
		result.append(f"{self.get_indent()}while {condition} do")

		self.indent()
		for stmt in node.body:
			result.append(self.visit(stmt))
		self.dedent()

		result.append(f"{self.get_indent()}end")

		return "\n".join(result)

	def visit_Break(self, node: ast.Break) -> str:
		"""Visit a break statement."""
		return f"{self.get_indent()}break"

	def visit_Continue(self, node: ast.Continue) -> str:
		"""Visit a continue statement."""
		# Lua doesn't have a direct equivalent to 'continue'
		return f"{self.get_indent()}-- continue is not directly supported in Lua"

	def visit_Pass(self, node: ast.Pass) -> str:
		"""Visit a pass statement."""
		return f"{self.get_indent()}-- pass"

	def visit_FunctionDef(self, node: ast.FunctionDef) -> str:
		"""Visit a function definition."""
		result = []

		if self.current_class:
			# Method definition
			func_name = f"{self.current_class}.{node.name}"

			# Check if this is an instance method (has self parameter)
			if node.args.args and node.args.args[0].arg == "self":
				params = [self.visit(arg) for arg in node.args.args[1:]]
				params_str = ", ".join(["self"] + params)
			else:
				params = [self.visit(arg) for arg in node.args.args]
				params_str = ", ".join(params)
		else:
			# Regular function definition
			func_name = node.name
			params = [self.visit(arg) for arg in node.args.args]
			params_str = ", ".join(params)

		result.append(f"{self.get_indent()}function {func_name}({params_str})")

		self.indent()
		for stmt in node.body:
			result.append(self.visit(stmt))
		self.dedent()

		result.append(f"{self.get_indent()}end")

		return "\n".join(result)

	def visit_Return(self, node: ast.Return) -> str:
		"""Visit a return statement."""
		if node.value:
			value = self.visit(node.value)
			return f"{self.get_indent()}return {value}"
		return f"{self.get_indent()}return"

	def visit_ClassDef(self, node: ast.ClassDef) -> str:
		"""Visit a class definition."""
		result = []

		class_name = node.name
		result.append(f"{self.get_indent()}{class_name} = {{}}")

		prev_class = self.current_class
		self.current_class = class_name

		# Visit class body
		for stmt in node.body:
			result.append(self.visit(stmt))

		# Add metatable for OOP behavior
		result.append(f"{self.get_indent()}{class_name}.__index = {class_name}")

		# Add constructor
		result.append(f"{self.get_indent()}function {class_name}.new(...)")
		result.append(f"{self.get_indent()}  local self = setmetatable({{}}, {class_name})")
		result.append(f"{self.get_indent()}  if {class_name}.__init then")
		result.append(f"{self.get_indent()}    {class_name}.__init(self, ...)")
		result.append(f"{self.get_indent()}  end")
		result.append(f"{self.get_indent()}  return self")
		result.append(f"{self.get_indent()}end")

		self.current_class = prev_class

		return "\n".join(result)

	def visit_Call(self, node: ast.Call) -> str:
		"""Visit a function call."""
		func = self.visit(node.func)
		args = [self.visit(arg) for arg in node.args]

		# Handle common built-in functions
		if isinstance(node.func, ast.Name):
			if node.func.id == "print":
				return f"print({', '.join(args)})"
			if node.func.id == "len":
				return f"#{args[0]}"
			if node.func.id == "str":
				return f"tostring({args[0]})"
			if node.func.id == "int":
				return f"math.floor({args[0]})"

		# Handle attribute access like obj.method()
		if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
			if node.func.value.id != "self" and node.func.attr == "append":
				# Handle list.append as table.insert
				return f"table.insert({self.visit(node.func.value)}, {', '.join(args)})"

		# Default function call syntax
		return f"{func}({', '.join(args)})"

	def visit_arg(self, node: ast.arg) -> str:
		"""Visit a function argument."""
		return node.arg

	def visit_List(self, node: ast.List) -> str:
		"""Visit a list literal."""
		elements = [self.visit(elt) for elt in node.elts]
		return f"{{{', '.join(elements)}}}"

	def visit_Dict(self, node: ast.Dict) -> str:
		"""Visit a dictionary literal."""
		items = []
		for i in range(len(node.keys)):
			key = self.visit(node.keys[i])
			value = self.visit(node.values[i])

			# Handle string keys differently
			if isinstance(node.keys[i], (ast.Str, ast.Constant)) and isinstance(
				getattr(node.keys[i], "s", getattr(node.keys[i], "value", None)),
				str,
			):
				# For string keys, use [key] = value syntax
				items.append(f"[{key}] = {value}")
			else:
				# For non-string keys, use [key] = value syntax
				items.append(f"[{key}] = {value}")

		return f"{{{', '.join(items)}}}"

	def visit_Tuple(self, node: ast.Tuple) -> str:
		"""Visit a tuple literal."""
		# Lua doesn't have tuples, so we represent them as tables
		elements = [self.visit(elt) for elt in node.elts]
		return f"{{{', '.join(elements)}}}"

	def visit_Subscript(self, node: ast.Subscript) -> str:
		"""Visit a subscript operation (indexing)."""
		value = self.visit(node.value)

		if isinstance(node.slice, ast.Index):
			# Python 3.8 and earlier
			idx = self.visit(node.slice.value)
		else:
			# Python 3.9+
			idx = self.visit(node.slice)

		# Adjust index for Lua (1-indexed)
		if isinstance(node.slice, (ast.Constant, ast.Num)) and isinstance(
			getattr(node.slice, "value", getattr(node.slice, "n", None)),
			int,
		):
			idx_val = getattr(node.slice, "value", getattr(node.slice, "n", 0))
			if idx_val >= 0:
				idx = str(idx_val + 1)

		return f"{value}[{idx}]"

	def visit_Attribute(self, node: ast.Attribute) -> str:
		"""Visit an attribute access."""
		value = self.visit(node.value)

		# Handle common special cases
		if isinstance(node.value, ast.Name):
			if node.value.id == "self" and self.current_class:
				return f"self.{node.attr}"

		return f"{value}.{node.attr}"

	def visit_Import(self, node: ast.Import) -> str:
		"""Visit an import statement."""
		result = []

		for name in node.names:
			module_name = name.name
			as_name = name.asname or module_name

			if module_name not in self.imported_modules:
				self.imported_modules.add(module_name)
				result.append(f'{self.get_indent()}local {as_name} = require("{module_name}")')

		return "\n".join(result)

	def visit_ImportFrom(self, node: ast.ImportFrom) -> str:
		"""Visit an import from statement."""
		result = []

		module_name = node.module

		if module_name not in self.imported_modules:
			self.imported_modules.add(module_name)
			result.append(f'{self.get_indent()}local {module_name} = require("{module_name}")')

		for name in node.names:
			import_name = name.name
			as_name = name.asname or import_name

			result.append(f"{self.get_indent()}local {as_name} = {module_name}.{import_name}")

		return "\n".join(result)

	def visit_Try(self, node: ast.Try) -> str:
		"""Visit a try/except statement."""
		result = []

		# In Lua, we use pcall for exception handling
		result.append(f"{self.get_indent()}local ok, err = pcall(function()")

		self.indent()
		for stmt in node.body:
			result.append(self.visit(stmt))
		self.dedent()

		result.append(f"{self.get_indent()}end)")

		if node.handlers:
			result.append(f"{self.get_indent()}if not ok then")
			self.indent()

			for handler in node.handlers:
				if handler.type:
					exc_type = self.visit(handler.type)
					if handler.name:
						exc_name = handler.name
						result.append(f"{self.get_indent()}local {exc_name} = err")

					result.append(f"{self.get_indent()}-- Exception type: {exc_type}")

				for stmt in handler.body:
					result.append(self.visit(stmt))

			self.dedent()
			result.append(f"{self.get_indent()}end")

		if node.finalbody:
			result.append(f"{self.get_indent()}-- finally")
			for stmt in node.finalbody:
				result.append(self.visit(stmt))

		return "\n".join(result)

	def visit_ListComp(self, node: ast.ListComp) -> str:
		"""Visit a list comprehension."""
		# Lua doesn't have list comprehensions, so we'll expand it to a loop
		result = []

		temp_var = f"temp_{id(node)}"
		result.append(f"{self.get_indent()}local {temp_var} = {{}}")

		generators = node.generators

		# Start with the outermost generator
		outer_target = self.visit(generators[0].target)
		outer_iter = self.visit(generators[0].iter)

		result.append(f"{self.get_indent()}for _, {outer_target} in ipairs({outer_iter}) do")
		self.indent()

		# Add any conditions from the outer generator
		for if_clause in generators[0].ifs:
			condition = self.visit(if_clause)
			result.append(f"{self.get_indent()}if {condition} then")
			self.indent()

		# Add the element to the result list
		element = self.visit(node.elt)
		result.append(f"{self.get_indent()}table.insert({temp_var}, {element})")

		# Close all the if statements from the outer generator
		for _ in generators[0].ifs:
			self.dedent()
			result.append(f"{self.get_indent()}end")

		self.dedent()
		result.append(f"{self.get_indent()}end")

		# Return the temporary variable
		result.append(f"{self.get_indent()}{temp_var}")

		return "\n".join(result)


def generate_runtime_helpers():
	"""Generate Lua code for runtime helpers to support Python-like behavior."""
	return """-- Python runtime helpers for Lua

-- Python-like 'in' operator for tables
function py_in(item, container)
    if type(container) == "string" then
        return string.find(container, item) ~= nil
    elseif type(container) == "table" then
        for _, v in ipairs(container) do
            if v == item then
                return true
            end
        end

        -- Check if it's a dictionary-like table
        if container[item] ~= nil then
            return true
        end
    end
    return false
end

-- Python-like 'range' function
function range(...)
    local args = {...}
    local start, stop, step

    if #args == 1 then
        start, stop, step = 0, args[1], 1
    elseif #args == 2 then
        start, stop, step = args[1], args[2], 1
    elseif #args == 3 then
        start, stop, step = args[1], args[2], args[3]
    else
        error("range() takes 1-3 arguments")
    end
    
    local t = {}
    for i = start, stop - 1, step do
        table.insert(t, i)
    end
    return t
end

-- Python-like 'enumerate' function
function enumerate(t, start)
    start = start or 0
    local result = {}
    for i, v in ipairs(t) do
        table.insert(result, {start + i - 1, v})
    end
    return result
end

-- Python-like 'zip' function
function zip(...)
    local args = {...}
    local result = {}
    local min_len = math.huge
    
    -- Find the minimum length among all arguments
    for _, t in ipairs(args) do
        min_len = math.min(min_len, #t)
    end
    
    -- Create tuples
    for i = 1, min_len do
        local tuple = {}
        for _, t in ipairs(args) do
            table.insert(tuple, t[i])
        end
        table.insert(result, tuple)
    end
    
    return result
end

-- Python-like string 'split' method
function string.split(s, sep)
    sep = sep or "%s"
    local t = {}
    for str in string.gmatch(s, "([^" .. sep .. "]+)") do
        table.insert(t, str)
    end
    return t
end
"""


def main():
	"""Main entry point for the transpiler CLI."""
	parser = argparse.ArgumentParser(description="Transpile Python code to Lua")
	parser.add_argument("input", help="Input Python file (use - for stdin)")
	parser.add_argument("-o", "--output", help="Output Lua file (default: stdout)")
	parser.add_argument("--runtime", action="store_true", help="Include runtime helpers in the output")

	args = parser.parse_args()

	# Read input code
	if args.input == "-":
		python_code = sys.stdin.read()
	else:
		with open(args.input) as f:
			python_code = f.read()

	# Transpile the code
	transpiler = LuaTranspiler()
	lua_code = transpiler.transpile(python_code)

	# Add runtime helpers if requested
	if args.runtime:
		runtime_helpers = generate_runtime_helpers()
		lua_code = f"{runtime_helpers}\n\n{lua_code}"

	# Write output
	if args.output:
		with open(args.output, "w") as f:
			f.write(lua_code)
	else:
		print(lua_code)


if __name__ == "__main__":
	main()
