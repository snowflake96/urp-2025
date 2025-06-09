import ansys.fluent.core as pyfluent

# ───────────────────────── 1. meshing ─────────────────────────
meshing = pyfluent.launch_fluent(mode="meshing", precision="double")
workflow = meshing.workflow
workflow.InitializeWorkflow(WorkflowType="Discretization Meshing")
tasks = workflow.TaskObject
print("TaskObject repr:", tasks)
print("TaskObject dir:", [attr for attr in dir(tasks) if not attr.startswith("_")])
task_names = tasks.getChildObjectDisplayNames()
print("Workflow task names:", task_names)
import sys; sys.exit(0)

# # 1-a) import the STL surface (faceted geometry)
# import_geom = tasks["Import Geometry"]
# import_geom.Arguments.set_state({"FileName": "aaaa.stl", "LengthUnit": "mm"})
# import_geom.Execute()

# # 1-b) generate the volume mesh directly
# tasks["Generate the Volume Mesh"].Execute()

# # 1-c) export a Fluent mesh
# meshing.tui.file.write_mesh("aaaa.msh")         # ASCII or .msh.h5
#                                                    # ─────────────────────────

# ──────────────────────── 2. solver & case ───────────────────────
solver = meshing.switch_to_solver()               # hands the mesh over
# (If you launched a fresh solver session you would instead do)
# solver.settings.file.read_mesh(file_name="artery.msh")

# 2-a) set or check boundary conditions here if you wish
# bc = solver.setup.boundary_conditions
# bc.velocity_inlet["inlet"].momentum.magnitude = 0.4  # etc.

# 2-b) write the final .cas.h5
solver.file.write_case(file_name="aaaa.cas.h5")  # ← conversion happens here
solver.exit()