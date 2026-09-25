"use client";
import { useEffect, useState } from "react";
import { adminListUsers, adminUpdateUser, adminVerifyExpert, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";

export default function AdminUsersPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [users, setUsers] = useState<any[]>([]);
  const [msg, setMsg] = useState("");
  const [form, setForm] = useState({ username:"", password:"", email:"", full_name:"", role:"doctor" });

  useEffect(()=>{ if(!loading && (!user || user.role!=="admin")) router.push("/"); },[user,loading]);

  const load = async () => {
    try { const data=await adminListUsers(); setUsers(data); } catch(e:any){ setMsg(e.message);} };
  useEffect(()=>{ load(); },[]);

  const createExpert = async () => {
    setMsg("");
    try {
      const res = await fetch(`${API_BASE}/v1/auth/register`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ username: form.username, email: form.email||undefined, password: form.password, full_name: form.full_name||undefined, role: form.role }), credentials:"include" });
      if(!res.ok) throw new Error(await res.text());
      setMsg("Created "+form.username);
      load();
    } catch(e:any){ setMsg(e.message); }
  };
  const toggleVerify = async (u:any) => {
    try { await adminUpdateUser(u.id, { is_verified: !u.is_verified }); load(); } catch(e:any){ setMsg(e.message); }
  };
  const verify = async (u:any)=>{ try{ await adminVerifyExpert(u.id); load(); }catch(e:any){ setMsg(e.message);} };
  const changeRole = async (u:any, role:string)=>{ try{ await adminUpdateUser(u.id, {role}); load(); }catch(e:any){ setMsg(e.message);} };

  if(loading) return <div>loading</div>;
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Quản trị User & Expert</h1>
      <p className="text-xs text-slate-600">Admin tạo/duyệt expert (doctor/pharmacist/specialist). Tự tạo admin qua CLI: python scripts/seed_admin.py</p>
      {msg && <div className="rounded-lg border bg-amber-50 p-2 text-xs">{msg}</div>}
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Tạo expert mới (admin)</h2>
        <div className="mt-3 grid gap-2 sm:grid-cols-5">
          <input placeholder="username" value={form.username} onChange={(e)=>setForm({...form, username:e.target.value})} className="rounded border px-2 py-1 text-xs" />
          <input placeholder="password" type="password" value={form.password} onChange={(e)=>setForm({...form, password:e.target.value})} className="rounded border px-2 py-1 text-xs" />
          <input placeholder="email" value={form.email} onChange={(e)=>setForm({...form, email:e.target.value})} className="rounded border px-2 py-1 text-xs" />
          <input placeholder="full name" value={form.full_name} onChange={(e)=>setForm({...form, full_name:e.target.value})} className="rounded border px-2 py-1 text-xs" />
          <select value={form.role} onChange={(e)=>setForm({...form, role:e.target.value})} className="rounded border px-2 py-1 text-xs">
            <option value="doctor">doctor</option><option value="pharmacist">pharmacist</option><option value="specialist">specialist</option><option value="admin">admin</option><option value="user">user</option>
          </select>
        </div>
        <button onClick={createExpert} className="mt-3 rounded-full bg-slate-900 text-white px-4 py-2 text-xs">Tạo</button>
      </div>
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Danh sách users ({users.length})</h2>
        <div className="mt-3 overflow-auto">
          <table className="w-full text-xs">
            <thead><tr className="border-b text-left"><th>User</th><th>Role</th><th>Verified</th><th>Active</th><th>Actions</th></tr></thead>
            <tbody>
              {users.map((u:any)=>(
                <tr key={u.id} className="border-b">
                  <td className="py-2">{u.username}<br/><span className="text-slate-500">{u.email}</span></td>
                  <td>
                    <select value={u.role} onChange={(e)=>changeRole(u, e.target.value)} className="rounded border px-1 py-1 text-xs">
                      <option value="user">user</option><option value="doctor">doctor</option><option value="pharmacist">pharmacist</option><option value="specialist">specialist</option><option value="admin">admin</option>
                    </select>
                  </td>
                  <td>{u.is_verified? "✅":"❌"}</td>
                  <td>{u.is_active? "●":"○"}</td>
                  <td className="flex gap-1 py-1">
                    {["doctor","pharmacist","specialist"].includes(u.role) && !u.is_verified && <button onClick={()=>verify(u)} className="rounded bg-emerald-600 text-white px-2 py-1">Verify</button>}
                    <button onClick={()=>toggleVerify(u)} className="rounded border px-2 py-1">{u.is_verified? "Unverify":"Verify"}</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
