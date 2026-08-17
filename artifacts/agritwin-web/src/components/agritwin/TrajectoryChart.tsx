import { useMemo } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { trajectoryData } from '@/data/demo';

export function TrajectoryChart() {
  const data = useMemo(() => trajectoryData, []);
  return (
    <div className="h-[280px] w-full" data-testid="chart-assimilation-trajectory">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 12, left: -10, bottom: 4 }}>
          <CartesianGrid stroke="#e9ece5" strokeDasharray="2 5" vertical={false} />
          <XAxis dataKey="day" tickLine={false} axisLine={{ stroke: '#dfe3dc' }} tick={{ fill: '#788078', fontSize: 10 }} tickFormatter={(value) => `D${value}`} />
          <YAxis domain={[0, 4]} tickLine={false} axisLine={false} tick={{ fill: '#788078', fontSize: 10 }} tickFormatter={(value) => value.toFixed(1)} />
          <Tooltip
            cursor={{ stroke: '#cbd7c7', strokeDasharray: '3 3' }}
            contentStyle={{ border: '1px solid #dfe3dc', borderRadius: 4, background: '#fff', fontFamily: 'Inter', fontSize: 11, boxShadow: '0 12px 30px rgba(27,48,34,.08)' }}
            formatter={(value: number | string, name: string) => [`${Number(value).toFixed(2)}`, name]}
            labelFormatter={(label) => `Simulation day ${label}`}
          />
          <Line type="monotone" dataKey="openLoop" name="Open-loop simulation" stroke="#1b3022" strokeWidth={2} dot={false} animationDuration={900} />
          <Line type="monotone" dataKey="assimilated" name="Assimilated simulation" stroke="#8fa382" strokeWidth={2} strokeDasharray="6 5" dot={false} animationDuration={1100} />
          <Line type="monotone" dataKey="observation" name="Observations" stroke="#d97706" strokeWidth={0} dot={{ r: 4, fill: '#d97706', stroke: '#fff', strokeWidth: 2 }} connectNulls={false} animationDuration={1200} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}